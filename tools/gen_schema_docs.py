# Copyright 2025 Digital Holography Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib, sys
from json_schema_for_humans.generate import generate_from_filename
from json_schema_for_humans.generation_configuration import GenerationConfiguration

SCHEMAS = pathlib.Path("src/holovibes/schemas/tasks")
OUTDIR  = pathlib.Path("doc/mkdocs/docs/schemas")
OUTDIR.mkdir(parents=True, exist_ok=True)

def cleanup():
    for f in OUTDIR.glob("*.md"):
        f.unlink()

def generate(schema: pathlib.Path):
    out = OUTDIR / f"{schema.stem}.md"
    print(f"Generating {out} from {schema}")
    config = GenerationConfiguration(
        template_name="md",
        custom_template_path="doc/md/base.md",
        description_is_markdown=True,
        link_to_reused_ref=True,
        copy_css=False,  # not needed for md
        copy_js=False,
        show_toc=False,
    )
    try:
        generate_from_filename(str(schema), str(out), config=config)
    except Exception as e:
        print(f"[ERROR] {schema}: {e}", file=sys.stderr)
        sys.exit(1)
    if not out.exists() or out.stat().st_size == 0:
        print(f"[ERROR] {schema}: generator returned but file missing/empty", file=sys.stderr)
        sys.exit(1)

def generate_all():
    for schema in SCHEMAS.rglob("*.json"):  # recurse
        generate(schema)

def main():
    cleanup()
    generate_all()

if __name__ == "__main__":
    main()
