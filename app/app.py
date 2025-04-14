# --- START OF FILE app.py ---

import gradio as gr
import os # <<< IMPORT OS
import torch
import cv2
import requests
import numpy as np
from pathlib import Path
import warnings
from basicsr.archs.rrdbnet_arch import RRDBNet
import time
import traceback
import webbrowser

# --- Configuration ---
APP_SCRIPT_PATH = Path(__file__).resolve()
APP_DIR = APP_SCRIPT_PATH.parent

# --- *** SET ENVIRONMENT VARIABLE *** ---
# Set BASICSR_HOME *before* importing gfpgan or its dependencies.
# This tells libraries like facexlib (used by gfpgan) where to store their downloaded weights.
os.environ['BASICSR_HOME'] = str(APP_DIR)
print(f"Set BASICSR_HOME to: {APP_DIR}")
# --- *** END SET ENVIRONMENT VARIABLE *** ---

# Now import libraries that might use BASICSR_HOME
import gfpgan
import realesrgan


# Suppress specific warnings if needed
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --- MODIFIED PATHS ---
MODEL_DIR = APP_DIR / "models"      # Main models directory relative to app directory
TEMP_DIR = APP_DIR / "temp"         # Temp directory relative to app directory
# Dependency weights (like face detectors) downloaded by gfpgan/facexlib
# should now go into APP_DIR/weights (e.g., app/weights) due to BASICSR_HOME.

# Create directories if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
# Note: The 'weights' directory (e.g., app/weights) will likely be created automatically
# by facexlib when it downloads the detection model, if it doesn't exist.

print(f"--- Path Configuration ---")
print(f"App Script Path: {APP_SCRIPT_PATH}")
print(f"App Directory (Script Location & BASICSR_HOME): {APP_DIR}")
print(f"Main Model Directory (inside app): {MODEL_DIR}")
print(f"Temp Directory (inside app): {TEMP_DIR}")
# print(f"Dependency Weights expected in: {APP_DIR / 'weights'}") # Optional print
print(f"--- End Path Configuration ---")

# --- Model Download (Main GFPGAN/RealESRGAN models) ---
MODEL_URLS = {
    "GFPGANv1.2.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth",
    "GFPGANv1.3.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "RestoreFormer.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
}

def download_model(url, model_path):
    """Downloads a model file if it doesn't exist."""
    if not model_path.exists():
        print(f"Downloading {model_path.name} to {model_path.parent}...")
        try:
            response = requests.get(url, stream=True, timeout=180)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {model_path.name} successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {model_path.name}: {e}")
            if model_path.exists():
                try: os.remove(model_path)
                except OSError as rm_err: print(f"Error removing partially downloaded file {model_path}: {rm_err}")
            raise
    else:
        print(f"{model_path.name} already exists in {model_path.parent}.")

print(f"--- Checking/Downloading Main Models (into {MODEL_DIR}) ---")
for model_name, url in MODEL_URLS.items():
    download_model(url, MODEL_DIR / model_name)
print("--- Main Model Check Complete ---")

# --- Model Initialization ---
if torch.cuda.is_available():
    device = 'cuda'; print("CUDA detected, using GPU.")
else:
    device = 'cpu'; print("CUDA not available, using CPU. Processing may be slow.")

half_precision = False
bg_upsampler = None
try:
    bg_model_path = MODEL_DIR / "RealESRGAN_x2plus.pth"
    if bg_model_path.exists():
        # Note: RealESRGANer initialization might also trigger downloads if its
        # specific model wasn't in MODEL_URLS and handled by download_model.
        # Setting BASICSR_HOME should cover this case too.
        bg_upsampler = realesrgan.RealESRGANer(
            scale=2, model_path=str(bg_model_path),
            model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
            device=device, tile=400, tile_pad=10, pre_pad=0, half=half_precision
        )
        print("Background upsampler (Real-ESRGAN) initialized.")
    else: print(f"Warning: RealESRGAN model not found at {bg_model_path}. Background upsampling disabled.")
except Exception as e:
    print(f"Error initializing Real-ESRGAN: {e}. Background upsampling will be unavailable."); traceback.print_exc(); bg_upsampler = None

# --- Restoration Function ---
# (No changes needed within restore_face itself for this path issue)
def restore_face(image, version, rescale_factor, progress=gr.Progress()):
    """Restores faces in the input image using the selected GFPGAN/RestoreFormer version."""
    if image is None: return None, None, "Status: Idle", "Please upload an image."

    print(f"\n--- Starting Restoration ---"); print(f"Version: {version}, Rescale Factor: {rescale_factor}")
    progress(0, desc="Starting restoration...")
    restorer = None; output_path_str = None

    try:
        # Determine model file and architecture based on version
        if version == 'v1.2': model_file, arch = "GFPGANv1.2.pth", 'clean'
        elif version == 'v1.3': model_file, arch = "GFPGANv1.3.pth", 'clean'
        elif version == 'v1.4': model_file, arch = "GFPGANv1.4.pth", 'clean'
        elif version == 'RestoreFormer': model_file, arch = "RestoreFormer.pth", 'RestoreFormer'
        else:
            error_msg = f"Error: Unknown version selected: {version}"
            print(error_msg)
            return None, None, "Status: Error", error_msg

        model_path = MODEL_DIR / model_file
        print(f"Using main model: {model_path}") # Clarified "main model"
        if not model_path.exists():
            error_msg = f"Error: Main model file not found: {model_path}. Please check the '{MODEL_DIR.name}' directory inside '{APP_DIR.name}'."
            print(error_msg)
            return None, None, "Status: Error", error_msg

        progress(0.1, desc="Loading model...")
        # Initializing GFPGANer is where dependency models (like face detectors)
        # are often checked and downloaded if missing. BASICSR_HOME should redirect these.
        current_bg_upsampler = bg_upsampler
        restorer = gfpgan.GFPGANer(
            model_path=str(model_path),
            upscale=int(rescale_factor),
            arch=arch,
            channel_multiplier=2,
            bg_upsampler=current_bg_upsampler,
            device=device
        )
        progress(0.2, desc="Model loaded.")

        input_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        progress(0.3, desc="Detecting/Restoring faces...")
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5
        )
        progress(0.8, desc="Faces restored, blending...")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_filename = f"output_{version.replace('.', '')}_{timestamp}.png"
        output_path = TEMP_DIR / base_filename
        output_path_str = str(output_path)

        if restored_img is not None:
            restored_img_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
            try:
                cv2.imwrite(output_path_str, restored_img)
                progress(1.0, desc="Completed")
                success_msg = f"Restoration successful. Output saved to: {output_path_str}"
                print(success_msg)
                print(f"--- Restoration Finished ---")
                return restored_img_rgb, output_path_str, "Status: Completed", success_msg
            except Exception as write_err:
                error_msg = f"Error writing output file: {write_err}"
                print(error_msg); traceback.print_exc()
                progress(1.0, desc="Write Error")
                return restored_img_rgb, None, "Status: Write Error", error_msg
        else:
            progress(1.0, desc="Failed/No Faces")
            input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            orig_filename = f"original_input_{timestamp}.png"
            output_path = TEMP_DIR / orig_filename
            output_path_str = str(output_path)
            warning_msg = "Warning: No faces found or restoration failed. Returning original image."
            try:
                cv2.imwrite(output_path_str, input_img) # Save original BGR
                print(f"{warning_msg} Saved original to: {output_path_str}")
            except Exception as write_err:
                print(f"Error writing original image file on failure: {write_err}")
                output_path_str = None
            print(f"--- Restoration Finished (No Faces/Failed) ---")
            return input_img_rgb, output_path_str, "Status: Warning", warning_msg

    except Exception as e:
        error_msg = f"An error occurred during restoration: {e}"
        print(error_msg); traceback.print_exc()
        progress(1.0, desc="Error")
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            error_filename = f"error_input_{timestamp}.png"
            output_path = TEMP_DIR / error_filename
            output_path_str = str(output_path)
            input_img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path_str, input_img_bgr)
            input_img_rgb = image
            print(f"Saving input image due to error: {output_path_str}")
            print(f"--- Restoration Finished (Error) ---")
            return input_img_rgb, output_path_str, "Status: Error", error_msg
        except Exception as e_fallback:
            fallback_error_msg = f"A critical error occurred during fallback saving: {e_fallback}"
            print(fallback_error_msg); traceback.print_exc()
            print(f"--- Restoration Finished (Critical Error) ---")
            return None, None, "Status: Critical Error", f"{error_msg}\n{fallback_error_msg}"
    finally:
        if restorer is not None:
            del restorer
            print("GFPGAN restorer object deleted.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")

DESCRIPTION = """
<center><h1>GFPGAN GRADIO Refactored by DrStr4Nge147</h1></center>

**Gradio demo for GFPGAN: Towards Real-World Blind Face Restoration with Generative Facial Prior.**

It can be used to restore your **old photos** or improve **AI-generated faces**.

<b style='color:red'>Note:</b> Output images are temporarily stored in the `app/temp` folder. Download results you want to keep.

To use it, simply upload your image, select the GFPGAN version and desired rescaling factor, then click **Submit**.
""" # Updated description slightly

# --- Build Gradio Blocks ---
# (No changes needed in the Gradio UI layout itself)
with gr.Blocks(css="footer {display: none !important}") as demo:
    gr.Markdown(DESCRIPTION)
    # ... (rest of Gradio layout remains the same) ...
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Input Image")
            version = gr.Radio(
                ["v1.2", "v1.3", "v1.4", "RestoreFormer"],
                label="GFPGAN Version",
                value="v1.4"
            )
            rescaling_factor = gr.Slider(
                label="Rescaling factor (Upscale)",
                minimum=1,
                maximum=8,
                step=1,
                value=2
            )
            with gr.Row():
                clear_button = gr.Button("Clear")
                submit_button = gr.Button("Submit", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(type="numpy", label="Output Image")
            output_download = gr.File(label="Download Restored Image")
            status_update = gr.Textbox(label="Status", value="Status: Idle", interactive=False, lines=1)
            info_message = gr.Textbox(label="Info/Error Message", value="", interactive=False, lines=2)

    # --- Button Actions ---
    submit_button.click(
        fn=restore_face,
        inputs=[input_image, version, rescaling_factor],
        outputs=[output_image, output_download, status_update, info_message],
        api_name="restore_face"
    )

    def clear_outputs():
        return None, "v1.4", 2, "Status: Idle", "", None, None

    clear_button.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[
            input_image, version, rescaling_factor,
            status_update, info_message, output_image, output_download
        ]
    )


# --- Launch App ---
if __name__ == "__main__":
    print("\n--- Starting Gradio App ---")
    # ... (rest of launch code remains the same) ...
    should_open_browser = True
    if os.environ.get('GRADIO_SERVER_NAME') or os.environ.get('GRADIO_SHARE'):
        should_open_browser = False

    if should_open_browser:
        print("Attempting to open browser at http://localhost:7860 ...")
        try:
            webbrowser.open("http://localhost:7860")
        except Exception as wb_err:
            print(f"Could not open browser automatically: {wb_err}")

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )
    print("--- Gradio App Execution Finished ---")

# --- END OF FILE app.py ---
