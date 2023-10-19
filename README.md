<!--
<p align="center">
  <img src="https://github.com//f-mac-tf/raw/main/docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
  f_mac_tf
</h1>

<p align="center">
    </a>  
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /> 
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
    <a href="https://github.com//f-mac-tf/blob/main/.github/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/>
    </a>
</p>

Implementation of F-MAC algorithm of Frantar et. al. (2021) in Tensorflow.

## :warning: Remark
We use absolute imports from the project home directory (e.g. here where the README is located).
For this the project home directory must be added to the module search path. This can for example be achieved by adding a '.env' file at project home level containing 'export PYTHONPATH="$PYTHONPATH:$PWD"'.

## 💪 Getting Started

Each experiment must be defined in a configuration in yaml syntax (which is read using ['hydra'](https://hydra.cc/docs/intro/)).
By calling `run_experiment(<configuration name>)` in 'run_experiments.py' one or multiple experiments can be run.
For each experiment and optimizer a dedicated folder will be created in 'logs' containing csv files of tracked metrics for each run and parameter configuration.

## 👋 Attribution
The idea of our optimizer(s) are not our own but based on the paper by Frantar et. al. 2021 [1].
Special thanks to Prof. Dr. David Rügamer for supervising us.

### ⚖️ License

The code in this package is licensed under the MIT License.

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.

## References

\[1\] GFrantar, Elias et al. “M-FAC: Efficient Matrix-Free Approximations of Second-Order Information.” Neural Information Processing Systems (2021).