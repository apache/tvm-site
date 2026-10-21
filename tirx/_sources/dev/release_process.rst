..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Release Process
===============

This guide describes the Apache TVM TIRx release workflow for creating a release
candidate, staging artifacts on ASF SVN, finalizing the release tag, and
publishing release artifacts.

.. admonition:: Prerequisite
   :class: hint

   The release manager sets the following variables before running these steps:

   .. code-block:: bash

      export TIRX_VERSION="v0.1.0-rc0"
      export TIRX_RELEASE_VERSION="v0.1.0"
      export ASF_USERNAME="your-apache-username"

   - Tools: ``git``, ``svn``, ``gpg``, ``gtar``, and ``shasum``.
   - An ASF account and a signing key published in the project's ``KEYS`` file.

.. important::

   These examples use GNU tar as ``gtar`` on macOS. Use ``tar`` on Linux.

Pre-release checklist (Step 0)
------------------------------

Before running the steps below, prepare the candidate tag and release thread.
Keep the package version consistent in ``pyproject.toml`` (``project.version``),
``CMakeLists.txt`` (``TVM_TIRX_VERSION``), and ``python/tvm_tirx/libinfo.py``
(``__version__``). Tags do not update these fields automatically.

**Step 0.1.** Tag the candidate ``$TIRX_VERSION`` on
`GitHub <https://github.com/apache/tvm-tirx/releases/new>`__ at the reviewed commit.
Use the intended package version, such as ``0.1.0``, in all three files.

**Step 0.2.** Open a `release tracking issue
<https://github.com/apache/tvm-tirx/issues/new>`__. Call the release vote once
the artifacts below are staged and verified.

Step 1. Create Release Candidate
--------------------------------

This builds and signs the source release artifacts under
``tvm-tirx-$TIRX_VERSION/``. Use a fresh release workspace:

.. code-block:: bash

   _make_tarball() {
     local _ver="$1"
     local _workdir="tvm-tirx-$_ver-release-files"
     local _release_dir="tvm-tirx-$_ver"
     local _source_dir="apache-tvm-tirx-$TIRX_RELEASE_VERSION"
     local _tarball="apache-tvm-tirx-src-$TIRX_RELEASE_VERSION.tar.gz"

     mkdir -p "$_workdir" "$_release_dir"
     (
       cd "$_workdir"
       git clone --recursive --branch "$_ver" https://github.com/apache/tvm-tirx.git "$_source_dir"
       gtar --exclude-vcs -czf "$_tarball" "$_source_dir"
       gpg --armor --output "$_tarball.asc" --detach-sig "$_tarball"
       shasum -a 512 "$_tarball" > "$_tarball.sha512"
       gpg --verify "$_tarball.asc" "$_tarball"
       shasum -a 512 -c "$_tarball.sha512"
     )
     mv "$_workdir/$_tarball" "$_workdir/$_tarball.asc" "$_workdir/$_tarball.sha512" "$_release_dir/"
   }

   _upload_to_apache_dev_svn() {
     local _ver="$1"
     local _release_dir="tvm-tirx-$_ver"
     svn checkout --depth=files https://dist.apache.org/repos/dist/dev/tvm svn-tvm-dev
     cp -R "$_release_dir" svn-tvm-dev/
     (
       cd svn-tvm-dev
       svn add "$_release_dir"
       svn commit --username "$ASF_USERNAME" -m "Add TIRx $_ver"
     )
   }

   _make_tarball "$TIRX_VERSION"
   _upload_to_apache_dev_svn "$TIRX_VERSION"

Download the staged archive, verify its signature and checksum, and build and test
the extracted sources using :doc:`source_build` and :doc:`ci_cd`. Check that
licenses and recursive submodule sources are included. The artifact uses its
final filename from the start so its checksum and signature survive promotion.

Call the vote on ``dev@tvm.apache.org`` with the candidate tag, commit, artifact
URL, signing-key information, and release notes. Follow the `ASF voting rules
<https://www.apache.org/legal/release-policy.html#release-approval>`__ and record
the result in the release thread.

Step 2. Conclude Release
------------------------

After the vote passes, conclude the vote, publish wheels, finalize the release
tag, and bump the development version.

**Step 2.1.** Announce the voting results by opening a ``[RESULT][VOTE]`` issue on
the `issue tracker <https://github.com/apache/tvm-tirx/issues/new>`__.
Issue notifications are mirrored to ``discuss-archive@tvm.apache.org``;
post the result to ``dev@tvm.apache.org`` as well.

**Step 2.2.** Publish PyPI wheels with the `publication workflow
<https://github.com/apache/tvm-tirx/actions/workflows/publish_wheel.yml>`__.
See :doc:`ci_cd` for how wheels are built with cibuildwheel.

**Step 2.3.** Tag the approved candidate commit as the final version:

.. code-block:: bash

   git clone -o upstream https://github.com/apache/tvm-tirx.git tvm-tirx-release
   (
     cd tvm-tirx-release
     git fetch upstream --tags
     git tag -a "$TIRX_RELEASE_VERSION" "${TIRX_VERSION}^{commit}" -m "Release $TIRX_RELEASE_VERSION"
     git push upstream "$TIRX_RELEASE_VERSION"
   )

Keep the candidate tag for traceability. Then create a version-bump branch and
set all three version fields from Step 0 to the next development version
(for example, ``0.1.1.dev0`` after ``0.1.0``):

.. code-block:: bash

   cd tvm-tirx-release
   git switch -c bump-version upstream/main
   "${EDITOR:-vi}" pyproject.toml CMakeLists.txt python/tvm_tirx/libinfo.py
   git add pyproject.toml CMakeLists.txt python/tvm_tirx/libinfo.py
   git commit -m "Bump development version after $TIRX_RELEASE_VERSION"
   git push -u upstream bump-version

Open a pull request for the version bump.

Step 3. Upload Release Artifacts
--------------------------------

After the release is final, copy the approved RC artifacts into ``dist/release``
under the final version directory, preserving the files unchanged:

.. code-block:: bash

   svn copy \
     "https://dist.apache.org/repos/dist/dev/tvm/tvm-tirx-$TIRX_VERSION" \
     "https://dist.apache.org/repos/dist/release/tvm/tvm-tirx-$TIRX_RELEASE_VERSION" \
     --username "$ASF_USERNAME" -m "Release TIRx $TIRX_RELEASE_VERSION"

Verify the public downloads, update the download page and release notes, and
announce the release to the community.
