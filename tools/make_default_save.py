#!/usr/bin/env python3
"""Pack a save directory into default-save.json for the web build.

  tools/make_default_save.py SAVE_DIR [default-save.json] [--note "text"]

SAVE_DIR is mapped onto the guest's /save one level down: a file at
SAVE_DIR/extdata/<id>/slot0 becomes /save/extdata/<id>/slot0. The page writes
the files once, on first load, when the game has no save yet (see web/app.js
seedDefaultSave). Stage it next to the web build, or deploy/ for stage.sh.
"""
import argparse, base64, json, os

ap = argparse.ArgumentParser()
ap.add_argument('dir')
ap.add_argument('out', nargs='?', default='default-save.json')
ap.add_argument('--note', default=None)
args = ap.parse_args()

files = {}
for dp, _, names in os.walk(args.dir):
    for n in names:
        p = os.path.join(dp, n)
        rel = os.path.relpath(p, args.dir).replace(os.sep, '/')
        files['/save/' + rel] = base64.b64encode(open(p, 'rb').read()).decode()

out = {'format': 'r3ds-save-1', 'files': files}
if args.note:
    out['note'] = args.note
json.dump(out, open(args.out, 'w'))
print(f'{args.out}: {len(files)} files', flush=True)
