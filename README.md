# TVM Project Homepage

## Serve Locally

```bash
./serve_local.sh
```

## Deployment

We use the script [scripts/task_deploy_asf_site.sh](scripts/task_deploy_asf_site.sh)
to generate and deploy content to the asf-site branch.

The docs folder is not part of the source,
they are built separately from the tvm's docs
and updated via [scripts/task_docs_update.sh](scripts/task_docs_update.sh)
