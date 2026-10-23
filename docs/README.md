# docs/

`index.html` is a standalone page documenting the timed encounter: the design
decisions, what is built, and what is still open. No build step, no
dependencies — open the file in a browser.

## Publishing it

Settings → Pages → Source: *Deploy from a branch*, branch `main`, folder
`/docs`. It then serves at `https://<owner>.github.io/<repo>/`.

**Before you enable that, know that GitHub Pages is public to the internet**
even when the repository is private, unless the plan supports private Pages.

The page is written to be safe to publish: it describes the assessment's
*structure* but deliberately omits the wording of the ten items, so a student
who finds it does not get the list they are graded against. Keep it that way
if you add to it — the items themselves live in `rubric.json`, which stays in
the repository.
