
# 1. Pixi Environment Guide

This workspace uses pixi, which is a package manager and the README provides instructions for setting up and using pixi, a powerful environment and package manager.

To use this, do at least two things
- Install pixi (see 'Installation' below)
- Execute command 'pixi install'

Now, running a python script can best be done with 'pixi python <file>.py
Also, installing a new conda pacakge with 'pixi add <conda package>
    and a pip package with 'pixi add --pypi <pypi package>' 

# More detailed description for use of pixi

## What is Pixi?

Pixi is a fast, modern environment manager that allows you to create reproducible environments for your projects. It's similar to conda but with improved performance and dependency resolution.

## Installation

```bash
# Install on macOS/Linux
curl -fsSL https://pixi.sh/install.sh | bash

# Install on Windows (PowerShell)
irm https://pixi.sh/install.ps1 | iex
```

Verify your installation:

```bash
pixi --version
```

## Basic Usage

### Install an existing environment (after downloading code from Github)
https://pixi.sh/latest/python/tutorial/#installation-pixi-install

```bash
pixi install
```

### Create a New Project

```bash
# Initialize a new pixi project
pixi init
```

### Add Dependencies

```bash
# Add packages
pixi add numpy pandas matplotlib

# Add specific versions
pixi add python=3.10 pytorch=2.0
```

### Activate Environment

```bash
# Enter the pixi environment
pixi shell
```

### Run Commands

```bash
# Run a command in the environment without activating
pixi run python script.py

# Execute custom tasks defined in pixi.toml
pixi run test
```

## Environment Configuration

The `pixi.toml` file defines your project configuration:

```toml
[project]
name = "my-project"
version = "0.1.0"

[tasks]
test = "pytest"
start = "python main.py"

[dependencies]
python = ">=3.9"
numpy = ">=1.24.0"
```

## Managing Environments

```bash
# List environments
pixi list

# Remove an environment
pixi remove

# Update packages
pixi update
```

## Troubleshooting

- **Package conflicts**: Check version compatibility in pixi.toml
- **Environment not found**: Ensure you're in the project directory
- **Activation issues**: Verify pixi is correctly installed and in PATH

## Additional Resources

- [Pixi Documentation](https://pixi.sh/docs)
- [GitHub Repository](https://github.com/prefix-dev/pixi)

# 2. IMAGE modelling

The country_to_regions.csv file that gives the model regions and included countries and image_region_numbers.csv that maps the region names and numbers can be found at:
https://github.com/imagepbl/geospatial-metadata/tree/main/mappings
