# simphox
Design and simulation module for photonics

## Installation

Install in your python environment using:

`pip install -e simphox`

You can then change `simphox` if necessary.
When importing `simphox`, you can now treat it as any other module.
No filepath setting necessary because `simphox` will be in your environment's `site-packages` directory.

For the AIM PDK imports, please save PDK files
in a separate folder of your choice (or in `simphox/aim_lib/`).
You will always specify these folders when using the PDK 
(see `simphox.gds.aim.AIMPhotonicsChip`). Please do not commit
these files to `simphox` as they tend to inflate contributions
(these are specified via a `.gitignore`).

## Git Workflow

### Adding a new feature branch

```
git pull # update local based on remote
git checkout develop # start branch from develop
git checkout -b feature/feature-branch-name
```

Do all work on branch. After your changes, from the root folder, execute the following:

```
git add . && git commit -m 'insert your commit message here'
```


### Rebasing and pull request

First you need to edit your commit history by "squashing" commits. 
You should be in your branch `feature/feature-branch-name`.
First look at your commit history to see how many commits you've made in your feature branch:

```
git log
```
Count the number of commits you've made and call that N.
Now, execute the following:

```
git rebase -i HEAD~N
```
Squash any insignificant commits (or all commits into a single commit if you like).
A good tutorial is provided 
[here](https://medium.com/@slamflipstrom/a-beginners-guide-to-squashing-commits-with-git-rebase-8185cf6e62ec).

Now, you should rebase on top of the `develop` branch by executing:
```
git rebase develop
```
You will need to resolve any conflicts that arise manually during this rebase process.

Now you will force-push this rebased branch using:
```
git push -f
```

Then you must submit a pull request using this [link](https://github.com/solgaardlab/simphox/pulls).

### Updating develop and master

The admin of this repository is responsible for updating `develop` (unstable release)
and `master` (stable release). 
This happens automatically once the admin approves pull request.

```
git checkout develop
git merge feature/feature-branch-name
```

To update master:
```
git checkout master
git merge develop
```

As a rule, only one designated admin should have permissions to do these steps.