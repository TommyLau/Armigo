# Repository and report privacy

Public source and documentation must be independent of any contributor's local
machine layout. Do not commit personal home paths, machine-specific workspace
locations, local file URLs, private repository locations, account identifiers,
credentials, or game-derived data.

Build and test instructions use repository-relative paths or explicitly supplied
inputs. Reference projects are optional study material, not required neighboring
directories. No private workspace is required to build the source or run the
synthetic tests.

Runtime inputs remain local and user-selected. Diagnostic output may describe
the input supplied by its user; review and redact it before sharing. Do not
upload ROMs, keys, saves, captures, game hashes, private paths, or account data in
bug reports. Use synthetic reproductions when possible.

The repository check rejects common personal-path forms, local file URLs, and
prohibited content types in indexed files. It cannot establish that every
possible secret or proprietary fact is absent. Review staged changes and public
reports as well.

Use a privacy-preserving commit email when desired. Review author and committer
metadata, all branches, and history before public publication; removing a value
from the latest files does not remove it from earlier commits. Historical
rewrites require explicit authorization and must preserve unrelated work.
