# Privacy and provider media transfer

`evaluate()` sends the source image and sampled frames from each generated
video to the VLM endpoint configured by the researcher. Run evaluation only
when you are authorized to transfer that media to that provider.

PAWBench returns structured outcome and trustworthiness judgments. It does not
put API credentials, raw provider request or response bodies, local media
paths, internal run identifiers, or private study data into returned judgment
rows or official metric output. Keep the local benchmark package, generated
videos, and any provider-side logs under your own applicable data controls.
