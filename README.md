<!--
SPDX-FileCopyrightText: Marcel Klehr <mklehr@gmx.net>
SPDX-License-Identifier: CC0-1.0
-->

# Text2Image Stable Diffusion
Text2Image provider using Stable Diffusion XL by Stability AI

The models run completely on your machine. No private data leaves your servers.

## Ethical AI Rating
### Rating: 🟢

Positive:
* the software for training and inference of this model is open source
* the trained model is freely available, and thus can be run on-premises
* the training data is freely available, making it possible to check or correct for bias or optimise the performance and CO2 usage.

Learn more about the Nextcloud Ethical AI Rating [in our blog](https://nextcloud.com/blog/nextcloud-ethical-ai-rating/).

## Install
 * Place this app in **nextcloud/apps/**

or 

 * Install from the Nextcloud appstore

After installing this app you will need to run:

```
$ php occ text2image_stablediffusion:download-models
```

## Building the app

The app can be built by using the provided Makefile by running:

    make

This requires the following things to be present:
* make
* which
* tar: for building the archive
* curl: used if phpunit and composer are not installed to fetch them from the web
* npm: for building and testing everything JS
