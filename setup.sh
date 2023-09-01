#!/bin/bash

# Read the YAML file (.secrets.yml) and get the values for aws_access_key_id and aws_secret_access_key
AWS_ACCESS_KEY_ID=$(yq .aws_access_key_id .secrets.yml)
AWS_SECRET_ACCESS_KEY=$(yq .aws_secret_access_key .secrets.yml)

# Check if the values are not empty
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
  echo "Error: aws_access_key_id or aws_secret_access_key not found in .secrets.yml"
  exit 1
fi

# Set the environment variables
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# Determine the current shell and set the appropriate configuration file
if [[ $SHELL == *"zsh"* ]]; then
  CONFIG_FILE=~/.zshrc
elif [[ $SHELL == *"bash"* ]]; then
  CONFIG_FILE=~/.bashrc
else
  echo "Error: Unsupported shell. Only zsh and bash are supported."
  exit 1
fi

# Append the export lines to the configuration file to set them permanently
echo "export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" >> $CONFIG_FILE
echo "export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY" >> $CONFIG_FILE

echo "Environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY have been set permanently in $CONFIG_FILE."