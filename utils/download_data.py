"""
Grab two small public-domain texts from Project Gutenberg:

• Alice in Wonderland
• Dr. Jekyll and Mr. Hyde

They are saved as:

datasets/alice/alice.txt
datasets/jekyll/jekyll.txt
"""

import urllib.request, pathlib, textwrap

BOOKS = [
    ("alice",  "https://www.gutenberg.org/files/11/11-0.txt"),
    ("jekyll", "https://www.gutenberg.org/files/43/43-0.txt"),
]

for name, url in BOOKS:
    out_dir  = pathlib.Path(f"../datasets/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{name}.txt"

    if out_file.exists():
        print(f"[✓] {name} already downloaded")
        continue

    print(f"[↓] Downloading {name} …")
    urllib.request.urlretrieve(url, out_file)

    # Optional: strip the Gutenberg license header/footer
    txt = out_file.read_text("utf-8", errors="ignore")
    start = txt.find("*** START OF")
    end   = txt.find("*** END OF")
    if start != -1 and end != -1:
        txt = txt[start:end]
        out_file.write_text(textwrap.dedent(txt))

print("\n✅  Done.  Launch the notebook next.")
