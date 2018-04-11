.. _add-module:

*********************************
Adding an external module to OGGM
*********************************

Thanks for helping us to make the model better!

There are two ways to add a functionality to OGGM:

1. The easiest (and recommended) way for small bugfixes or simple functionalities
   is to add it directly to the main OGGM codebase. In this case, refer to
   the :ref:`contributing` page for how to do that.
2. If your endeavor is a larger project (i.e. a fundamental change in the model
   physics or a new workflow), you can still use option 1 above. However,
   there might be reasons (explained below) to use a different approach. This
   is what this page is for.

Why would I add my model to the OGGM set of tools?
==================================================

We envision OGGM as a **modular tool** to model glacier behavior. We do not
want to force anyone towards a certain set-up or paramaterization that we chose.
However, we believe that the OGGM capabilities could be useful to a
wide range of applications, and also to you.

Finally, we strongly believe in **model intercomparisons**. Agreeing on a
common set of boundary conditions is the first step towards meaningful
comparisons: this is where OGGM can help.

Can I use my own repository/website and my own code style to do this?
=====================================================================

Yes you can!

Ideally, we would have your module added to the main codebase: this ensures
consistency of the code and continuous testing. However, there are several
reasons why making your own repository could be useful:

- complying to OGGM's strict testing rules can be annoying, especially in the
  early stages of development of a model
- with you own repository you have full control on it's development while
  still being able to use the OGGM workflow and multiprocessing capabilities
- with an external module the users who will use your model *will* have
  to download it from here and you can make sure that correct attribution
  is made to your work, i.e. by specifying that using this module requires a
  reference to a specific scientific publication
- if your funding agency requires you to have your own public website
- etc.


To help you in this project we have set-up a
`template repository <https://github.com/OGGM/dummy-module>`_ from which you can
build upon.

Before writing your own module we recommend to contact us to discuss
the best path to follow in your specific case.
