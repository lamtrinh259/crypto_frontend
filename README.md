# Front end for cryptocurrencies prices forecasting
- Document here the project: crypto_frontend
- Description: Repository for the front-end of the cryptocurrencies prices forecasting
- Data Source: from Bitfinex exchange
- Type of project: using ML algorithms to forecast future prices of crypto
- Technology: Streamlit with Python

Below is some guidance regarding starting the project if you're interested in cloning this repo. 

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for crypto_frontend in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/crypto_frontend`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "crypto_frontend"
git remote add origin git@github.com:{group}/crypto_frontend.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
crypto_frontend-run
```

# Install

Go to `https://github.com/{group}/crypto_frontend` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/crypto_frontend.git
cd crypto_frontend
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
crypto_frontend-run
```
