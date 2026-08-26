# openwiki/

Generated documentation tree for coding agents, produced by
[OpenWiki](https://docs.langchain.com/oss/openwiki/overview) 0.4.0.
GitHub Pages publishes from `docs/` using the existing `mkdocs.yml` nav.
This directory is not a Wiki tab and has no `/wiki/` URL.

`INSTRUCTIONS.md` is the user-authored brief. OpenWiki reads it on every run
and does not overwrite it. Public pages use the same paths as `mkdocs.yml`
(`index.md`, `getting-started/`, `concepts/`, `configuration/`,
`integrations/`, `advanced/`). `testing/` is agent-only and is not copied to
`docs/`.

Generated pages, Grounded Claims under `.claims/`, and `.last-update.json`
appear after the `OpenWiki Update` GitHub Action runs with repository secret
`OPENROUTER_API_KEY`. After a successful update, `scripts/publish_openwiki_to_docs.py`
copies public paths into `docs/`.

Coding agents: read `INSTRUCTIONS.md` for invariants. Once pages exist, start
there instead of rediscovering the tree.
