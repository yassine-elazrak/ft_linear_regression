PROJECT_NAME	= ft_linear_regression


# -f is not used because --env-file is not available on the version of docker-compose
# the VM uses.

NEW_FILE		= touch
RM				= rm -rf

all:
		# source  /goinfre/yelazrak/ft_linear_regression/venv/bin/activate
		python srcs/main.py

install:
		pip install -r requirements.txt
		
# .up:
#   	source  venv/bin/activate
      

# .PHONY: install up