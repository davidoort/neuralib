To update your conda environment so that nnlib dependencies are met, run:
`conda env update --file environment.yml --prune
`

To update the environment.yml file with new dependencies, run:
`conda env update --prefix ./env --file environment.yml  --prune`


To update the requirements.txt file (for pip environments), run: `pip freeze > requirements.txt`