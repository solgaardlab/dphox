![dphox](https://user-images.githubusercontent.com/7623867/93381618-ca48ed00-f815-11ea-980c-0fff994441a5.png)

Design module for photonic simulations and tapeouts. 

**Note**: This is a work in progress.
No documentation or examples are yet available, but will be made available in the coming months.
Testing code is also yet to come, and will also be added in the coming months.

## Installation

Install in your python environment using:

`pip install -e dphox`

You can then change `dphox` if necessary.
When importing `dphox`, you can now treat it as any other module.
No filepath setting necessary because `dphox` will be in your environment's `site-packages` directory.

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
git push --set-upstream origin feature/feature-branch-name
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