# Artifact security and operational scope

Only task-owned research files, evidence and reports are selected for the portable package. Model weights, Python environments, hidden runtime directories, SSH control sockets and credentials are excluded. The package does not include another copy of the 100 supplied PDFs; their unchanged download manifest and individual review ledger are included.

Before final assembly, a generic text scan for password assignments, bearer-token headers, private-key blocks, Hugging Face token shapes, AWS access-key shapes and password-bearing SSH utilities found no matches in the authored experiment, literature, notes, log and artifact trees. This is a useful check, not a guarantee that every possible secret encoding has been detected. The final ZIP content manifest is checked separately against the packaged bytes.

Authentication was used only for the authorized university HPC host. Compute and installation ran under Slurm allocations; no privileged installation, binary GPU instrumentation, external security target, AWS cluster change, application submission or external message was performed. Task-owned HPC evidence remains available for the user. Connection cleanup does not delete remote experiments or authentication history from this conversation.

The password shared in the conversation should be rotated. It is intentionally not repeated in any research artifact or reproduction command.
