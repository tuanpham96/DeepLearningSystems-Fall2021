# HW3 - Transformers

## Description of HW

> Carefully read the [instructions](https://docs.google.com/document/d/1TeRDGWtfSYa08N30R4jJLboLub11WYzT/edit?usp=sharing&ouid=102957549596160300860&rtpof=true&sd=true)
> The template notebook is [here](https://drive.google.com/file/d/1hUMNe7vSa798z8pdXv6zzoKFg14lx_iX/view?usp=sharing)
>Organize the write-ups neatly in the provided sections.

See [Instructions](Instructions.pdf)

## Description of the folder organization

Anything for the main English-Spanish translations would have `en-es` or `spa` in their names. In the `output` folder, these are prefixed by `transformer-*.pkl`.

Anything for English-Vietnamese translations would have `en-vi`/`en-vn` or `vie` or `vn` in their names. In the `output` folder, these are prefixed by `engviet-transformer-*.pkl`.

Data and experiment preparations are in notebooks prefixed by [`notebooks/experiment-setup-en*`](notebooks). The results are saved in `data` folder for the nonbreaking prefixes files and training data files. Also, the model configurations are defined in these notebooks and saved in [`model/config/`](model/config/).

Because Colab needed attending to and has low limit, the models were run on Kaggle GPU instead, the resulting notebooks are in [`notebooks/kaggle-notebooks/`](notebooks/kaggle-notebooks/).

The results containing the losses and accuracies are saved and downloaded to [`output`](output). No checkpoints are downloaded in this repository as they are too big.

The reason for splitting was to anticipate Kaggle kernel timeout (I think around 9 hours or so for each continuous run). Additionally, model 10 (reduce vocab size) for EN-ES was not the right implementation when first run, so run again after update.
