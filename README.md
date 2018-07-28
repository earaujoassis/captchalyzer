# Captchalyzer

> Extractor of captcha data from analyzed and classified sound files

## Setup & Running

First you need to create and to activate the `venv` and install the requirements. Please make sure
to use Python 3 (it must be version 3.6.5 or below; `hmmlearn` is not installable through versions
above 3.6.6). You may use [`asdf`](https://github.com/asdf-vm/asdf) to manage Python versions; a
`.tool-versions` file is already provided.

```sh
$ python3 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

Audio files should be available at `audiofiles/base_treinamento_I` and `audiofiles/base_validacao_I`.

Once those requirements are met, you may run:

```sh
$ python captchalyzer.py
```

## License

[MIT License](http://earaujoassis.mit-license.org/) &copy; Ewerton Assis, for code base.
[CC BY-NC-ND](https://creativecommons.org/licenses/by-nc-nd/4.0/deed) &copy; Ewerton Assis, for the report.
