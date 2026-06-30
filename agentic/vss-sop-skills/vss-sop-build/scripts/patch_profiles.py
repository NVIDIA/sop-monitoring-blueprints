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

# Robust Profile Patching Script for VSS SOP Build
import sys
import os
import re

def patch_file_profiles(file_path, services_to_patch=None, profile_to_add="bp_sop_2d"):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return False

    with open(file_path, 'r') as f:
        content = f.read()

    lines = content.splitlines()
    new_lines = []
    
    current_service = None
    in_services_block = False

    for line in lines:
        stripped = line.strip()
        
        # Track if we are inside 'services:' block
        if line.startswith('services:'):
            in_services_block = True
            current_service = None
            new_lines.append(line)
            continue
            
        if in_services_block:
            # Check indentation to see if we exited services block (indent == 0 and not empty and not comment)
            leading_spaces = len(line) - len(line.lstrip())
            if leading_spaces == 0 and stripped and not stripped.startswith('#') and not line.startswith('services:'):
                in_services_block = False
                current_service = None
            elif leading_spaces == 2 and stripped.endswith(':'):
                current_service = stripped[:-1].strip()
            elif leading_spaces <= 2 and stripped and not stripped.startswith('#'):
                current_service = None

        # Clean MINIMAL_PROFILE suffix if any
        if '${MINIMAL_PROFILE:+_extended}' in line:
            line = line.replace('${MINIMAL_PROFILE:+_extended}', '')

        # Check if we need to patch this line
        if re.search(r'^\s+profiles:\s*\[', line):
            # If services_to_patch is specified, only patch if current_service is in the list
            if services_to_patch is None or (current_service and current_service in services_to_patch):
                m = re.search(r'profiles:\s*\[(.*)\]', line)
                if m:
                    inside = m.group(1)
                    # Split and clean items
                    items = [item.strip().strip('"').strip("'") for item in inside.split(',') if item.strip()]
                    
                    # Ensure profile_to_add is the first entry
                    if profile_to_add not in items:
                        items.insert(0, profile_to_add)
                    else:
                        items.remove(profile_to_add)
                        items.insert(0, profile_to_add)
                        
                    formatted_items = ", ".join(f'"{item}"' for item in items)
                    indent = line[:line.find('profiles:')]
                    line = f'{indent}profiles: [{formatted_items}]'
                    print(f"Patched profiles for service '{current_service}' in {os.path.basename(file_path)}")

        new_lines.append(line)

    with open(file_path, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: patch_profiles.py <yaml_file_path> [services_comma_separated] [profile_to_add]")
        sys.exit(1)
        
    file_path = sys.argv[1]
    services = None
    if len(sys.argv) > 2 and sys.argv[2].strip() and sys.argv[2] != "all":
        services = [s.strip() for s in sys.argv[2].split(',')]
        
    profile = "bp_sop_2d"
    if len(sys.argv) > 3:
        profile = sys.argv[3]
        
    patch_file_profiles(file_path, services, profile)

