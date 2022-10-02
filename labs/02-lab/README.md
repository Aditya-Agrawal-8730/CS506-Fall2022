# cs506

This is an example python package for implementing and re-using tools learned in BU CS506

## Setup

You need to have python3 and [pip installed](https://www.makeuseof.com/tag/install-pip-for-python/) on your laptop. If you are using windows, please take a look at [this resource](https://docs.microsoft.com/en-us/windows/python/beginners) for an example set up (terminal, git, IDE etc.).

Before you get started, please check your python3 version and change the line 3 of the `tox.ini` file accordingly. For example, if you are using python 3.8, the line should be:

```
envlist = py38
```

Additionally, I recommend installing [virtualenv](https://pypi.org/project/virtualenv/) (pip install this package) to manage the python packages you install for each project you create.

## Before you Start

Go to your local `CS506-Fall2022/labs/02-lab/` folder.

### Optional

Install virtualenv

```bash
    pip install virtualenv
```

Create a virtualenv in this folder (for windows users, please see https://programwithus.com/learn-to-code/Pip-and-virtualenv-on-Windows/ for the corresponding commands)

```bash
    virtualenv -p python3 <name-of-your-virtual-env>
```

Activate the virtualenv on Windows:

```bash
    source <name-of-your-virtual-env>/Scripts/activate
```

Otherwise:

```bash
    source <name-of-your-virtual-env>/bin/activate
```

(to deactivate the environment, just type "deactivate" in your terminal/powershell)

### Required

Install tox

```bash
    pip install tox
```

### Verify your setup

Run the tests with the following command

```bash
    tox 
```

Ensure that all the tests are failing because of a "NotImplementedError" being raised. Here is the expected last few lines of the error message from tox you should expect:

```
====================================== 11 failed, 1 passed in 0.18s ======================================
ERROR: InvocationError for command CS506-Fall2022/labs/02-lab/.tox/py39/bin/pytest --cov=cs506 (exited with code 1)
________________________________________________ summary ________________________________________________
ERROR:   py39: commands failed
```

## Goal

Take a look at the library functions [here](https://github.com/gallettilance/CS506-Fall2022/blob/master/labs/02-lab/cs506/read.py), [here](https://github.com/gallettilance/CS506-Fall2022/blob/master/labs/02-lab/cs506/sim.py)

Remove the

```python
raise NotImplementedError()
```

line and replace it with code that does what the function should do. Test your implementation by running "tox". The goal is to get all tests to pass. Upload your implementations to github by creating a pull request.

## Bonus

It's a good idea to take a look at the tests defined [here](https://github.com/gallettilance/CS506-Fall2022/blob/master/labs/02-lab/tests/test_read.py), [here](https://github.com/gallettilance/CS506-Fall2022/blob/master/labs/02-lab/tests/test_sim.py).

Contributions are always welcome (and encouraged) to clarify existing documentation in this repository!
