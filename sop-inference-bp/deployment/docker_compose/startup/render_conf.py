# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import argparse
import string
import logging
import os
import pprint

_LOGGER = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate configuration(s) from template(s)")
    parser.add_argument(
        "--output_file", type=str, required=True, action="append",
        help="Output file path for generated config. Can be specified multiple times, must match number of --template_file."
    )
    parser.add_argument(
        "--template_file", type=str, required=True, action="append",
        help="Template file path for config. Can be specified multiple times, must match number of --output_file."
    )
    parser.add_argument(
        "--log_level", type=str, required=True,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Logging level"
    )
    args = parser.parse_args()

    if len(args.output_file) != len(args.template_file):
        parser.error("The number of --output_file and --template_file arguments must be equal (one-to-one mapping).")

    output_files = args.output_file
    template_files = args.template_file
    log_level = args.log_level
    logging.basicConfig(level=log_level)

    # Process each template file and output file pair in order
    for template_file, output_file in zip(template_files, output_files):
        _LOGGER.info(f"Processing template: {template_file} -> output: {output_file}")

        with open(template_file, "r") as fp:
            template_content = fp.read()

        template = string.Template(template_content)

        # FIXME: This is dangerous, we should have not ever print environment variables.
        #_LOGGER.debug("Environment variables:")
        #_LOGGER.debug(pprint.pformat(dict(os.environ)))
        rendered_content = template.safe_substitute(os.environ)
        _LOGGER.info("Rendered content:")
        _LOGGER.info(rendered_content)

        with open(output_file, "w") as fp:
            fp.write(rendered_content)

if __name__ == "__main__":
    main()
