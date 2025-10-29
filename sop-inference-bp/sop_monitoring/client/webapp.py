# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
SOP Monitoring Web Application

A Gradio-based web interface for uploading videos and running SOP monitoring inference
with real-time streaming responses displayed in a chat-bot style interface.
"""

import json
import time
import os
from typing import Generator

import gradio as gr
import requests
from openai import OpenAI

# This service is also launched by the docker compose. It should be able to connect to the API server.
# raise if this variable is not set
API_SERVER_NAME = os.environ["API_SERVER_NAME"]
API_SERVER_PORT = os.environ["API_SERVER_PORT"]
API_BASE_URL = f"http://{API_SERVER_NAME}:{API_SERVER_PORT}/v1"
UI_ROOT_PATH = os.environ["DEMO_WEB_APP_ROOT_PATH"]
#API_BASE_URL = "http://localhost:8080/v1"

THIS_SERVICE_PORT = int(os.environ["DEMO_WEB_APP_PORT"])

class SOPMonitoringClient:
    """Client for interacting with SOP Monitoring API using OpenAI client"""

    def __init__(self, base_url: str):
        self.client = OpenAI(
            base_url=base_url,
            api_key="dummy_key",  # API key not needed for local server
            max_retries=1
        )
        self.base_url = base_url

    def upload_file(self, file_path: str, purpose: str = "vision") -> str | None:
        """Upload a file and return the file ID"""
        try:
            with open(file_path, "rb") as f:
                uploaded_file = self.client.files.create(
                    file=f,
                    purpose=purpose
                )
                return uploaded_file.id
        except Exception as e:
            print(f"Upload error: {e}")
            return None

    def stream_chat_completion(self, file_id: str, prompt: str) -> Generator[str, None, None]:
        """Stream chat completion responses"""
        try:
            response = self.client.chat.completions.create(
                model="nvila-8b-assy17",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_file",
                                "image_file": {
                                    "file_id": file_id
                                }
                            }
                        ]
                    }
                ],
                stream=True
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"❌ Streaming error: {e}"

    def detect_sop(self, action_json: str, vlm_output: str, checker_id: str = "*", keep_alive: bool = True) -> dict:
        """Call SOP detection endpoint"""
        try:
            sop_detection_request = {
                "action_json": action_json,
                "vlm_output": vlm_output,
                "keep_alive": keep_alive,
                "checker_id": checker_id,
            }

            response = requests.post(
                f"{self.base_url}/sop/detection",
                json=sop_detection_request,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"SOP detection failed: {response.status_code} {response.text}"}
        except Exception as e:
            return {"error": f"SOP detection error: {e}"}


# Initialize the API client
api_client = SOPMonitoringClient(API_BASE_URL)


def load_action_json(action_file) -> tuple[dict, str]:
    """Load and validate action JSON file"""
    if action_file is None:
        return None, "❌ Please upload an action JSON file."

    try:
        with open(action_file, 'r') as f:
            action_config = json.load(f)

        # Validate required fields
        if "actions" not in action_config:
            return None, "❌ Invalid action JSON: missing 'actions' field."

        return action_config, f"✅ Action JSON loaded successfully! Found {len(action_config['actions'])} actions."
    except json.JSONDecodeError as e:
        return None, f"❌ Invalid JSON format: {e}"
    except Exception as e:
        return None, f"❌ Error loading action file: {e}"


def generate_sop_prompt(action_config: dict) -> str:
    """Generate the proper VLM prompt format for SOP monitoring"""
    if not action_config or "actions" not in action_config:
        return "Please upload an action JSON file first."

    actions = action_config["actions"]
    prefix = f"There are {len(actions)-1} possible actions.\nWhat actions does the operator take?"

    prompt_lines = [prefix] + actions
    return "\n".join(prompt_lines)


def stream_inference(video_file, action_file, prompt_text, history):
    """Run streaming inference with SOP detection and show individual chat messages for each result"""
    # Validation checks
    if video_file is None:
        history.append({"role": "user", "content": prompt_text})
        history.append({"role": "assistant", "content": "❌ Please upload a video file first."})
        return history, prompt_text

    if action_file is None:
        history.append({"role": "user", "content": prompt_text})
        history.append({"role": "assistant", "content": "❌ Please upload an action JSON file first."})
        return history, prompt_text

    if not prompt_text.strip() or prompt_text == "Upload an action JSON file to see the generated prompt...":
        history.append({"role": "user", "content": ""})
        history.append({"role": "assistant", "content": "❌ Please upload an action JSON file to generate a prompt."})
        return history, prompt_text

    try:
        # Load action configuration
        action_config, load_msg = load_action_json(action_file)
        if action_config is None:
            history.append({"role": "user", "content": prompt_text})
            history.append({"role": "assistant", "content": load_msg})
            return history, prompt_text

        # Upload the file to API server
        file_id = api_client.upload_file(video_file, "vision")
        if not file_id:
            history.append({"role": "user", "content": prompt_text})
            history.append({"role": "assistant", "content": "❌ Failed to upload video file."})
            return history, prompt_text

        # Initialize analysis
        history.append({"role": "user", "content": prompt_text})
        history.append({"role": "assistant", "content": "🚀 **Starting Analysis...** I'll show each detected action and its SOP analysis as separate messages."})
        yield history, prompt_text

        # Initialize SOP detection
        action_json_str = json.dumps(action_config)
        checker_id = "*"
        cycle = 0
        missing_detected = []
        misordered_detected = []
        final_missing_detected = []
        final_misordered_detected = []
        cycle_completed = False
        summary = None
        action_count = 0

        # Stream the response with SOP detection
        for chunk in api_client.stream_chat_completion(file_id, prompt_text):
            # Run SOP detection on non-empty chunks
            if chunk.strip():
                action_count += 1

                # Add VLM output as a separate chat message
                vlm_message = f"🆕 **Action Count {action_count}:** {chunk.strip()}"
                history.append({"role": "assistant", "content": vlm_message})
                yield history, prompt_text
                time.sleep(0.1)

                sop_result = api_client.detect_sop(
                    action_json=action_json_str,
                    vlm_output=chunk,
                    checker_id=checker_id,
                    keep_alive=True
                )

                if "error" not in sop_result:
                    checker_id = sop_result.get("checker_id", checker_id)
                    new_cycle = sop_result.get("cycle", cycle)
                    missing_detected = sop_result.get("missing_detected", missing_detected)
                    misordered_detected = sop_result.get("misordered_detected", misordered_detected)
                    cycle_completed = sop_result.get("cycle_completed", cycle_completed)

                    # Add SOP detection results as a separate chat message
                    sop_msg = f"🔍 **SOP Analysis for Action Count {action_count}:**\n"
                    sop_msg += f"- **Cycle:** {new_cycle}"
                    if new_cycle != cycle:
                        sop_msg += f" *(changed from {cycle})*"
                        cycle = new_cycle

                    # Add colored missing actions
                    if missing_detected:
                        missing_str = f"<span style='color: #DC2626; font-weight: bold;'>{missing_detected}</span>"
                    else:
                        missing_str = "<span style='color: #16A34A;'>None</span>"
                    sop_msg += f"\n- **Missing Actions:** {missing_str}"

                    # Add colored misordered actions
                    if misordered_detected:
                        misordered_str = f"<span style='color: #EA580C; font-weight: bold;'>{misordered_detected}</span>"
                    else:
                        misordered_str = "<span style='color: #16A34A;'>None</span>"
                    sop_msg += f"\n- **Misordered Actions:** {misordered_str}"

                    sop_msg += f"\n- **Cycle Completed:** {'✅ Yes' if cycle_completed else '❌ No'}"

                    # Add status indicator
                    if missing_detected or misordered_detected:
                        sop_msg = "⚠️ " + sop_msg
                    else:
                        sop_msg = "✅ " + sop_msg

                    history.append({"role": "assistant", "content": sop_msg})
                else:
                    error_msg = f"❌ **SOP Detection Error for Action Count {action_count}:** {sop_result['error']}"
                    history.append({"role": "assistant", "content": error_msg})

                yield history, prompt_text
                time.sleep(0.1)

        # Finalize SOP detection
        if checker_id != "*":
            final_sop_result = api_client.detect_sop(
                action_json=action_json_str,
                vlm_output="",
                checker_id=checker_id,
                keep_alive=False
            )

            if "error" not in final_sop_result:
                final_missing_detected = final_sop_result.get("final_missing_detected", final_missing_detected)
                final_misordered_detected = final_sop_result.get("final_misordered_detected", final_misordered_detected)
                final_cycle_completed = final_sop_result.get("cycle_completed", cycle_completed)
                final_summary = final_sop_result.get("summary", summary)

                # Add final summary as a separate chat message
                final_msg = f"🏁 **Final SOP Analysis Summary:**\n"
                final_msg += f"- **Total Actions Count:** {action_count}\n"
                final_msg += f"- **Final Cycle Count:** {cycle}\n"
                # Add colored final missing actions
                if final_missing_detected:
                    final_missing_str = f"<span style='color: #DC2626; font-weight: bold;'>{final_missing_detected}</span>"
                else:
                    final_missing_str = "<span style='color: #16A34A;'>None</span>"
                final_msg += f"- **Missing Actions:** {final_missing_str}\n"

                # Add colored final misordered actions
                if final_misordered_detected:
                    final_misordered_str = f"<span style='color: #EA580C; font-weight: bold;'>{final_misordered_detected}</span>"
                else:
                    final_misordered_str = "<span style='color: #16A34A;'>None</span>"
                final_msg += f"- **Misordered Actions:** {final_misordered_str}\n"

                final_msg += f"- **Cycle Completed:** {'✅ Yes' if final_cycle_completed else '❌ No'}"
                if final_summary:

                    final_msg += f"\n- **Summary:**\n<div style='background-color: #F0F9FF; padding: 10px; border-radius: 6px; border-left: 4px solid #0EA5E9;'>\n```json\n{json.dumps(final_summary, indent=2)}\n```\n</div>"

                # Add overall status
                if final_missing_detected or final_misordered_detected:
                    final_msg = "⚠️ " + final_msg + "\n\n*Issues detected in SOP compliance.*"
                else:
                    final_msg = "✅ " + final_msg + "\n\n*All actions completed correctly according to SOP.*"

                history.append({"role": "assistant", "content": final_msg})
                yield history, prompt_text
            else:
                error_msg = f"❌ **Final SOP Detection Error:** {final_sop_result['error']}"
                history.append({"role": "assistant", "content": error_msg})
                yield history, prompt_text

    except Exception as e:
        history.append({"role": "user", "content": prompt_text})
        history.append({"role": "assistant", "content": f"❌ **Error during analysis:** {e}"})
        yield history, prompt_text


# Create the Gradio interface
with gr.Blocks(
    title="SOP Monitoring System",
    theme=gr.themes.Soft(),
    css="""
    .video-container {
        max-height: 350px !important;
        height: 350px !important;
        overflow: hidden;
    }
    .video-container video {
        max-height: 300px !important;
        width: 100% !important;
        object-fit: contain;
    }
    .chat-container { height: 600px; }
    """
) as demo:

    gr.Markdown("# 🔍 SOP Monitoring System")
    gr.Markdown("Upload a video and analyze Standard Operating Procedures with real-time AI inference")

    with gr.Row():
                # Left column - Video upload and preview
        with gr.Column(scale=1):
            gr.Markdown("### 📹 Upload Files")

            with gr.Row():
                video_input = gr.File(
                    label="Upload MP4 Video",
                    file_types=[".mp4", ".avi", ".mov", ".mkv"],
                    type="filepath",
                    scale=1
                )

                action_file_input = gr.File(
                    label="Upload Action JSON",
                    file_types=[".json"],
                    type="filepath",
                    scale=1
                )

            video_preview = gr.Video(
                label="Video Preview",
                elem_classes=["video-container"],
                height=350,
                show_download_button=False
            )

            combined_status = gr.Textbox(
                label="Upload Status",
                interactive=False,
                max_lines=3,
                placeholder="Upload video and action JSON files..."
            )

            # Simple status update function
            def update_combined_status(video_file, action_file):
                status_lines = []

                # Video status
                if video_file is not None:
                    try:
                        file_id = api_client.upload_file(video_file, "vision")
                        if file_id:
                            status_lines.append(f"✅ Video uploaded! File ID: {file_id}")
                        else:
                            status_lines.append("❌ Video upload failed")
                    except Exception as e:
                        status_lines.append(f"❌ Video error: {e}")
                else:
                    status_lines.append("📹 Please upload a video file")

                # Action JSON status
                if action_file is not None:
                    try:
                        config, action_msg = load_action_json(action_file)
                        status_lines.append(action_msg)
                    except Exception as e:
                        status_lines.append(f"❌ Action JSON error: {e}")
                else:
                    status_lines.append("📋 Please upload an action JSON file")

                return "\n".join(status_lines)

            # Update video preview when video is uploaded
            video_input.change(
                fn=lambda v: v,
                inputs=[video_input],
                outputs=[video_preview]
            )

            # Update status when files change
            video_input.change(
                fn=update_combined_status,
                inputs=[video_input, action_file_input],
                outputs=[combined_status]
            )

            action_file_input.change(
                fn=update_combined_status,
                inputs=[video_input, action_file_input],
                outputs=[combined_status]
            )

        # Right column - Chat interface
        with gr.Column(scale=1):
            gr.Markdown("### 💬 AI Analysis Chat")

            chatbot = gr.Chatbot(
                label="SOP Analysis Results",
                elem_classes=["chat-container"],
                height=600,
                type="messages",
                placeholder="Upload a video and start chatting to analyze SOPs..."
            )

            prompt_input = gr.Textbox(
                label="Generated SOP Prompt",
                placeholder="Upload an action JSON file to see the generated prompt...",
                lines=8,
                interactive=True,
                max_lines=15
            )

            with gr.Row():
                send_btn = gr.Button("🚀 Analyze", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)

    # Auto-populate prompt when action JSON is uploaded
    def auto_populate_prompt(action_file):
        if action_file is not None:
            try:
                config, _ = load_action_json(action_file)
                if config:
                    return generate_sop_prompt(config)
                else:
                    return "Error loading action JSON file."
            except Exception as e:
                return f"Error processing action JSON: {e}"
        return "Upload an action JSON file to see the generated prompt..."

    action_file_input.change(
        fn=auto_populate_prompt,
        inputs=[action_file_input],
        outputs=[prompt_input]
    )

    # Event handlers
    send_btn.click(
        fn=stream_inference,
        inputs=[video_preview, action_file_input, prompt_input, chatbot],
        outputs=[chatbot, prompt_input],
        show_progress=True
    )

    clear_btn.click(
        fn=lambda current_prompt: ([], current_prompt),
        inputs=[prompt_input],
        outputs=[chatbot, prompt_input]
    )

    # Instructions
    gr.Markdown("### 💡 How to Use")
    gr.Markdown("""
    1. **Upload Files**: Upload both a video file and action JSON file
    2. **Auto-Generated Prompt**: The SOP prompt will be automatically generated from your action JSON
    3. **Analyze**: Click the Analyze button to start real-time SOP monitoring
    4. **Results**: See streaming inference with live action detection and SOP compliance status
    5. **Video**: Use the video player controls to play, pause, or seek through your video as needed

    *Note: You may occasionally see Gradio warnings about file uploads - these are harmless and can be ignored.*
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=THIS_SERVICE_PORT,
        share=False,
        show_error=False,
        debug=False,
        root_path=UI_ROOT_PATH,
    )
