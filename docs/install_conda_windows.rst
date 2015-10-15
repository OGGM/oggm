Install checklist on a Windows system with conda
================================================

This is a template for Johannes

1. Install conda
----------------

Do this and this


2. git and GitHub
--------------------

We need git::

    $ sudo apt-get install git
    $ git config --global user.name "John Doe"
    $ git config --global user.email johndoe@example.com

And we need to get a SSH key for not having to retype a password all the time.
Details: https://help.github.com/articles/generating-ssh-keys/

Once you added the key to GitHub, clone the repository where you want::

    $ git clone https://github.com/OGGM/oggm


X. Python Packages
------------------

Be sure to be on the working environment::

    $ activate oggm


5. Testing
----------

It's easier with nose::

    $ conda install nose

And in oggm's root directory::

    $ nosetests
