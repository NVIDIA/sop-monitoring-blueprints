#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import gi
import argparse
import os
import time
from datetime import datetime

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
gi.require_version('GstPbutils', '1.0')
gi.require_version('GstSdp', '1.0')
gi.require_version('GstRtsp', '1.0')
from gi.repository import Gst, GstRtspServer, GstRtsp, GLib, GstPbutils, GstSdp

class VideoRtspMediaFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, filename, mode: str = "passthrough"):
        GstRtspServer.RTSPMediaFactory.__init__(self)
        self.filename = os.path.abspath(filename)
        self.mode = mode
        self.framerate = None
        self.video_codec = None
        self._discover_stream_info(self.filename)

    def _discover_stream_info(self, filename):
        try:
            discoverer = GstPbutils.Discoverer()
            uri = "file://" + filename
            info = discoverer.discover_uri(uri)
            video_streams = info.get_video_streams()
            if video_streams:
                stream = video_streams[0]
                num = stream.get_framerate_num()
                denom = stream.get_framerate_denom()
                if num > 0 and denom > 0:
                     print(f"Discovered framerate: {num}/{denom}")
                     self.framerate = f"{num}/{denom}"

                caps = stream.get_caps()
                if caps and caps.get_size() > 0:
                    codec_name = caps.get_structure(0).get_name()
                    self.video_codec = codec_name
                    print(f"Discovered video codec: {codec_name}")
        except Exception as e:
            print(f"Error discovering stream info: {e}")

    def do_configure(self, rtsp_media):
        rtsp_media.set_reusable(True)
        rtsp_media.set_latency(200)

    def on_src_probe(self, pad, info):
        buf = info.get_buffer()
        if buf:
            if not hasattr(self, 'buf_count'):
                self.buf_count = 0
            self.buf_count += 1
            if self.buf_count % 100 == 0:
                pts = buf.pts if buf.pts != Gst.CLOCK_TIME_NONE else -1
                print(f"Pushed buffer: pts={pts}", flush=True)
        return Gst.PadProbeReturn.OK

    def _get_passthrough_parser_and_payloader(self):
        """Return (parser, payloader) elements based on detected video codec."""
        codec = self.video_codec or ""
        if "h265" in codec or "x-h265" in codec:
            return "h265parse", "rtph265pay name=pay0 pt=96 config-interval=1"
        elif "mpeg" in codec and "h264" not in codec and "h265" not in codec:
            return "mpeg4videoparse", "rtpmp4vpay name=pay0 pt=96 config-interval=1"
        elif "h264" in codec or "x-h264" in codec or not codec:
            return "h264parse", "rtph264pay name=pay0 pt=96 config-interval=1"
        else:
            print(f"Warning: unknown codec '{codec}', falling back to decodebin path")
            return None, None

    def do_create_element(self, url):
        if self.mode == "passthrough":
            parser, payloader = self._get_passthrough_parser_and_payloader()
            if parser is None:
                print(f"Codec '{self.video_codec}' not supported in passthrough, falling back to overlay mode")
                return self._create_overlay_pipeline()

            pipeline_str = (
                f'filesrc location="{self.filename}" ! '
                'qtdemux name=demux '
                f'demux.video_0 ! '
                f'{parser} ! '
                'identity name=loop_identity single-segment=true ! '
                f'{payloader}'
            )
            print(f"Creating pipeline in passthrough mode (codec={self.video_codec}): {pipeline_str}")
            try:
                bin_element = Gst.parse_launch(pipeline_str)
                identity_el = bin_element.get_by_name("loop_identity")
                if identity_el:
                    pad = identity_el.get_static_pad("sink")
                    if pad:
                        print("Adding EOS/Flush probe to identity sink pad for looping", flush=True)
                        pad.add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, self.on_eos_probe, bin_element)
                
                pay_el = bin_element.get_by_name("pay0")
                if pay_el:
                    srcpad = pay_el.get_static_pad("src")
                    if srcpad:
                        srcpad.add_probe(Gst.PadProbeType.QUERY_DOWNSTREAM | Gst.PadProbeType.QUERY_UPSTREAM, self.on_query_probe)
                        srcpad.add_probe(Gst.PadProbeType.BUFFER, self.on_src_probe)
                return bin_element
            except Exception as e:
                print(f"Error creating passthrough pipeline: {e}")
                return None

        return self._create_overlay_pipeline()

    def _create_overlay_pipeline(self):
        # Host-side H.264 encoder detection. This mirrors the encoder-fallback logic in
        # vss-sop-deploy/scripts/verify_rtsp_components.py and the RTSP output generated
        # from ../ds-sop-skills/deepstream-sop/references/skill_18_rtsp_streaming_output.md.
        # These run in isolated runtimes and cannot share an import, so keep the encoder
        # preference order consistent.
        encoder_name = None
        if Gst.ElementFactory.find("nvh264enc"):
            encoder_name = "nvh264enc"
        elif Gst.ElementFactory.find("nvv4l2h264enc"):
            encoder_name = "nvv4l2h264enc"
        elif Gst.ElementFactory.find("x264enc"):
            encoder_name = "x264enc"
        elif Gst.ElementFactory.find("openh264enc"):
            encoder_name = "openh264enc"
            
        pipeline_str = ""
        
        framerate_caps = ""
        if self.framerate:
             framerate_caps = f",framerate={self.framerate}"
        
        common_pre = (
            f'filesrc location="{self.filename}" ! '
            'decodebin ! '
            'videorate ! '
            f'video/x-raw{framerate_caps} ! '
            'videoscale ! '
            'video/x-raw ! '
            'videoconvert name=logger ! '
            'textoverlay name=clock valignment=top halignment=right font-desc="Sans 10" shaded-background=true ! '
        )
        
        post_enc_caps = f"video/x-h264{framerate_caps} ! " if self.framerate else ""
        
        if encoder_name == "nvh264enc":
            print(f"Creating pipeline with NVIDIA Hardware Encoder ({encoder_name})")
            pipeline_str = (
                f'{common_pre}'
                'videoconvert ! '
                f'video/x-raw{framerate_caps} ! '
                'nvh264enc bitrate=2048 rc-mode=vbr ! '
                f'{post_enc_caps}'
                'identity name=loop_identity single-segment=true ! '
                'rtph264pay name=pay0 pt=96 config-interval=1'
            )
        elif encoder_name == "nvv4l2h264enc":
            print(f"Creating pipeline with NVIDIA V4L2 Hardware Encoder ({encoder_name})")
            pipeline_str = (
                f'{common_pre}'
                'nvvideoconvert nvbuf-memory-type=2 compute-hw=1 ! '
                'video/x-raw(memory:NVMM) ! '
                'nvv4l2h264enc bitrate=4000000 iframeinterval=10 ! '
                f'{post_enc_caps}'
                'identity name=loop_identity single-segment=true ! '
                'rtph264pay name=pay0 pt=96 config-interval=1'
            )
        elif encoder_name == "x264enc":
            print(f"Creating pipeline with Software Encoder ({encoder_name})")
            pipeline_str = (
                f'{common_pre}'
                'videoconvert ! '
                'x264enc speed-preset=fast tune=zerolatency bitrate=4000 ! '
                f'{post_enc_caps}'
                'identity name=loop_identity single-segment=true ! '
                'rtph264pay name=pay0 pt=96'
            )
        elif encoder_name == "openh264enc":
            print(f"Creating pipeline with OpenH264 Encoder ({encoder_name})")
            pipeline_str = (
                f'{common_pre}'
                'videoconvert ! '
                'openh264enc bitrate=4000000 ! '
                f'{post_enc_caps}'
                'identity name=loop_identity single-segment=true ! '
                'rtph264pay name=pay0 pt=96'
            )
        else:
            print("Warning: No suitable H.264 encoder found.")
            pipeline_str = (
                f'{common_pre}'
                'videoconvert ! '
                'x264enc ! '
                f'{post_enc_caps}'
                'identity name=loop_identity single-segment=true ! '
                'rtph264pay name=pay0 pt=96'
            )
        
        print(f"Pipeline string: {pipeline_str}")
        try:
            bin_element = Gst.parse_launch(pipeline_str)
            logger_el = bin_element.get_by_name("logger")
            if logger_el:
                pad = logger_el.get_static_pad("src")
                if pad:
                    pad.add_probe(Gst.PadProbeType.BUFFER, self.log_timestamp)
            
            clock_el = bin_element.get_by_name("clock")
            if clock_el:
                print("Found clock element, adding probe.")
                pad = clock_el.get_static_pad("video_sink")
                if pad:
                    pad.add_probe(Gst.PadProbeType.BUFFER, self.update_clock, clock_el)
                else:
                    print("Could not find 'video_sink' pad for clock element. Trying 'sink'...")
                    pad = clock_el.get_static_pad("sink")
                    if pad:
                         pad.add_probe(Gst.PadProbeType.BUFFER, self.update_clock, clock_el)
                    else:
                         print("Could not find 'sink' pad either. Pads available:")
                         for p in clock_el.pads:
                             print(f"  - {p.get_name()}")
            else:
                print("Could not find clock element")

            identity_el = bin_element.get_by_name("loop_identity")
            if identity_el:
                pad = identity_el.get_static_pad("sink")
                if pad:
                     print("Adding EOS/Flush probe to identity sink pad for looping", flush=True)
                     pad.add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, self.on_eos_probe, bin_element)
            
            pay_el = bin_element.get_by_name("pay0")
            if pay_el:
                srcpad = pay_el.get_static_pad("src")
                if srcpad:
                    srcpad.add_probe(Gst.PadProbeType.QUERY_DOWNSTREAM | Gst.PadProbeType.QUERY_UPSTREAM, self.on_query_probe)
                    srcpad.add_probe(Gst.PadProbeType.BUFFER, self.on_src_probe)
                    
            return bin_element
        except Exception as e:
            print(f"Error creating pipeline: {e}")
            return None

    def on_query_probe(self, pad, info):
        query = info.get_query()
        if query.type == Gst.QueryType.DURATION:
            # Tell the server this stream has infinite duration (live)
            query.set_duration(Gst.Format.TIME, -1)
            return Gst.PadProbeReturn.HANDLED
        return Gst.PadProbeReturn.OK

    def on_eos_probe(self, pad, info, bin_element):
        event = info.get_event()
        if event.type == Gst.EventType.EOS:
            print("EOS detected, restarting stream...", flush=True)
            GLib.idle_add(self.seek_to_start, bin_element)
            return Gst.PadProbeReturn.DROP
        elif event.type == Gst.EventType.FLUSH_START:
            print("Dropping FLUSH_START", flush=True)
            return Gst.PadProbeReturn.DROP
        elif event.type == Gst.EventType.FLUSH_STOP:
            print("Dropping FLUSH_STOP", flush=True)
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    def seek_to_start(self, bin_element):
        print("Seeking to start...", flush=True)
        # Flush is required for demuxer to seek properly, but we drop flushes before payloader
        res = bin_element.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0)
        if not res:
            print("Seek failed!", flush=True)
        return False

    def log_timestamp(self, pad, info):
        buf = info.get_buffer()
        if buf and buf.pts != Gst.CLOCK_TIME_NONE:
            # Print timestamp every 30 frames (approx)
            if buf.pts % 1000000000 < 40000000:
                 print(f"SERVER: PTS={buf.pts} WallTime={time.time():.3f}", flush=True)
        return Gst.PadProbeReturn.OK

    def update_clock(self, pad, info, clock_el):
        now = datetime.now()
        # Format: YYYY-MM-DD HH:MM:SS.mmm
        time_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        # print(f"Updating clock to: {time_str}") 
        clock_el.set_property("text", time_str)
        return Gst.PadProbeReturn.OK

class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, filename, port=8554, mount_point="/test", mode: str = "passthrough"):
        super(GstServer, self).__init__()
        self.set_service(str(port))
        
        self.factory = VideoRtspMediaFactory(filename, mode=mode)
        self.factory.set_shared(True)
        self.connect("client-connected", self.on_client_connected)
        
        mount_points = self.get_mount_points()
        mount_points.add_factory(mount_point, self.factory)
        
        print(f"RTSP stream ready at rtsp://127.0.0.1:{port}{mount_point}")

    @staticmethod
    def _inject_framerate_into_sdp(sdp_text: str, framerate: str) -> str:
        """
        Insert framerate into SDP in multiple ways:
        - a=framerate / a=x-framerate (common but not always consumed)
        - append framerate=... into the H264 fmtp line (some parsers map fmtp params into caps)
        We keep the exact fraction form (e.g. 301/12) because some downstream tools
        prefer it and it's unambiguous.
        """
        lines = sdp_text.splitlines()
        out = []

        in_video = False
        video_has_fps = False
        video_fmtp_patched = False

        def _maybe_add_fps():
            nonlocal video_has_fps, video_fmtp_patched
            if in_video and not video_has_fps:
                out.append(f"a=framerate:{framerate}")
                out.append(f"a=x-framerate:{framerate}")
            video_has_fps = False
            video_fmtp_patched = False

        for line in lines:
            if line.startswith("m="):
                _maybe_add_fps()
                in_video = line.startswith("m=video")
            if in_video and (line.startswith("a=framerate:") or line.startswith("a=x-framerate:")):
                video_has_fps = True
            if in_video and line.startswith("a=fmtp:") and "framerate=" not in line:
                # Append as a custom fmtp parameter (non-standard).
                out.append(f"{line};framerate={framerate}")
                video_fmtp_patched = True
                continue
            out.append(line)

        _maybe_add_fps()
        return "\r\n".join(out) + "\r\n"

    def on_client_connected(self, server, client):
        # Patch outgoing DESCRIBE SDP to include framerate attributes.
        client.connect("send-message", self.on_client_send_message)

    def on_client_send_message(self, client, ctx, response):
        try:
            # Force live stream by replacing Range header with npt=now-
            res, range_val = response.get_header(GstRtsp.RTSPHeaderField.RANGE, 0)
            if res == GstRtsp.RTSPResult.OK and range_val:
                response.remove_header(GstRtsp.RTSPHeaderField.RANGE, 0)
                response.add_header(GstRtsp.RTSPHeaderField.RANGE, "npt=now-")

            res, body_list = response.get_body()
            if res != GstRtsp.RTSPResult.OK or not body_list:
                return
            sdp_text = bytes(body_list).decode("utf-8", errors="ignore")
            # Only DESCRIBE responses normally carry SDP
            if "v=0" not in sdp_text or "m=video" not in sdp_text:
                return

            # Patch SDP to remove a=range and add framerate
            lines = sdp_text.splitlines()
            out = []
            for line in lines:
                if line.startswith("a=range:"):
                    out.append("a=range:npt=now-")
                else:
                    out.append(line)
            patched = "\r\n".join(out) + "\r\n"

            if self.factory.framerate:
                patched = self._inject_framerate_into_sdp(patched, self.factory.framerate)
            
            if patched != sdp_text:
                response.set_body(list(patched.encode("utf-8")))
                # print(f"Patched SDP with framerate and live range", flush=True)
        except Exception as e:
            print(f"SDP patching error: {e}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Create RTSP stream from video file with clock overlay")
    parser.add_argument("--filename", help="Path to video file")
    parser.add_argument("--port", default=8552, type=int, help="RTSP server port (default: 8552)")
    parser.add_argument("--mount", default="/sensor_0", help="RTSP mount point (default: /test)")
    parser.add_argument(
        "--mode",
        default="passthrough",
        choices=["passthrough", "overlay"],
        help="passthrough: stream original H.264 from MP4 (preserves framerate); overlay: decode+overlay+re-encode (may lose framerate)",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filename):
        print(f"Error: File {args.filename} not found.")
        sys.exit(1)
        
    Gst.init(None)
    
    mount_point = args.mount
    if mount_point and not mount_point.startswith("/"):
        mount_point = "/" + mount_point
    
    server = GstServer(args.filename, args.port, mount_point, mode=args.mode)
    server.attach(None)
    
    loop = GLib.MainLoop()
    try:
        print("Starting server... Press Ctrl+C to stop.")
        loop.run()
    except KeyboardInterrupt:
        print("\nStopping server...")
        pass

if __name__ == '__main__':
    main()

