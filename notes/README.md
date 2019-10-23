`yxg/notes/paper` is a submodule of the core repo `yxg`.

It has been set up to link and sync changes in the local repo with Overleaf,
where the project `paper` is hosted. In this way, we avoid pushing the entire repo to Overleaf every time, which contains files too large for Overleaf to handle.


## Set-up

In order to make any changes to the paper, you should set up the link locally as a submodule. To do that, first clean up (or move to an outside backup directory) `yxg/notes/paper/`.

1. For example (running from the repo root folder `yxg/`):
  - `$ mv notes/paper/ ~/Desktop`
  - `$ git rm -r --cached notes/paper`

2. Then, again from the root folder run:
  - `$ git submodule add --name paper https://git.overleaf.com/5bb5f3f767b04d11c24c5c9d notes/paper` creates a submodule called 'paper' in `notes/paper`, setting the Overleaf project as a remote.

This will clone the contents of the Overleaf remote locally.

## Push changes

1. Navigate to `yxg/notes/paper`. This is a submodule with Overleaf set as remote. Push the changes to Overleaf.

* From the repo root directory:
  - `$ cd notes/paper`
  - `$ git add <file>`
  - `$ git commit -m "<message>"`
  - `$ git push origin master`

2. (Optional Step): Also push to GitHub. It is a good idea because others can see something has changed in the Overleaf remote before they start editing.

* From the repo root directory:
  - `$ git add notes/paper`
  - `$ git commit -m "<message>"`
  - `$ git push origin master`


## Pull changes

Same principle as PUSH-ing changes. You can pull the `yxg/notes/paper` submodule individually by navigating there and pulling, or you can go to the root directory of the repo and do `$ git pull` to pull all remotes in the repo tree.

* (One step) from the repo root directory:
  - `$ git pull`

* (Two steps) from the repo root directory:
  - `$ git pull origin master`
* This will pull the GitHub remote. Then:
  - `$ cd notes/paper`
  - `$ git pull origin master`

This will pull the Overleaf remote.

## General Notes

* Overleaf is set to also automatically sync all of its contents to Dropbox. The link is https://www.dropbox.com/sh/du0v4mac7xewgjp/AABBAGbumom11CVUP2VVO_qla.
* This is a 2-way sync, meaning, any changes to Dropbox will also update the Overleaf remote repo.`
