# Captchalyzer

> Extractor of captcha data from analyzed and classified sound files

## Setup & Running

First you need to create and to activate the `venv` and install the requirements. Please
make sure to use Python 3 (version 3.7 at least).

```sh
$ python3 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

Audio files should be available at `audiofiles/base_treinamento_I` and `audiofiles/base_validacao_I`.

Once these requirements are met, you may run:

```sh
$ python captchalyzer.py
```


## License

[MIT License](http://earaujoassis.mit-license.org/) &copy; Ewerton Assis
