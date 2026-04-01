echo clean Jupyter Notebook outputs in all .ipynb files in the current directory and subdirectories
@echo off
for /r %%f in (*.ipynb) do (
    echo Clearing: %%f
    nbstripout "%%f"
)