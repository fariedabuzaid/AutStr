.PHONY: media media-automaton media-gource

# Regenerate the images committed under docs/media/ and embedded in the README.
# Needs: graphviz, gource, ffmpeg, gifsicle, and a display (headless machines are
# handled automatically via xvfb by the gource script).
media: media-automaton media-gource

# The automaton diagram (fast).
media-automaton:
	python scripts/gen_readme_media.py

# The gource history GIF (slow; renders the whole commit history).
media-gource:
	bash scripts/gen_gource_gif.sh
