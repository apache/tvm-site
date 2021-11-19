# TVM Project Homepage

## Dependencies

1. Install Ruby

    ```bash
    # for Ubuntu 18.04+
    sudo apt install -y ruby-full build-essential
    ```

2. Install Jekyll and Bundler gems

    ```bash
    gem install bundler jekyll
    ```

3. Install project dependencies

    ```bash
    git clone https://github.com/apache/tvm-site.git
    cd tvm-site

    # If this runs into resolution errors, you may need to run:
    # bundle config set --local path 'vendor/cache'
    bundle install
    ```

## Serve Locally

```bash
./serve_local.sh

# If you are developing on a remote machine, you can set up an SSH tunnel to view
# the site locally:
ssh -L 4000:localhost:4000 <the remote>
```

Then visit [http://localhost:4000](http://localhost:4000) in your browser.

## Deployment

We use the script [scripts/task_deploy_asf_site.sh](scripts/task_deploy_asf_site.sh)
to generate and deploy content to the asf-site branch.

The docs folder is not part of the source,
they are built separately from the tvm's docs
and updated via [scripts/task_docs_update.sh](scripts/task_docs_update.sh)
