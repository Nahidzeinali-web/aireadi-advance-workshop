# Hosting the notebook for a group

One web address and one shared password. Everyone who logs in gets their own copy of the notebook and the sample
data, running in a separate container on the host machine. Participants install nothing.

## For participants

1. Open the workshop address in a browser.
2. Type any name (your first name is fine; two people using the same name would share one copy) and the workshop
   password.
3. The notebook opens. Run each cell with **Shift + Enter**.

Your copy is yours alone and is saved as you work. A server that sits idle for two hours is stopped; log in again
with the same name and your copy is still there.

## For the host

Requirements: Docker with the compose plugin, Tailscale with Funnel enabled, the `data` folder next to the notebook
(unzip `aireadi_workshop_data.zip` in the repository folder), and a fast drive for the workshop files.

Workshop files live in one folder, by default `/mnt/fast/aireadi-workshop` (set `WORKSHOP_STORAGE` in `hosting/.env`
to use another). Put it on the fastest drive available, not the boot drive:

| Folder | Contents |
|---|---|
| `data/` | the sample data, mounted read-only into every participant's server |
| `template/` | the notebook each person gets on first login |
| `users/<name>/` | one folder per participant, holding their copy of the notebook and anything they save |

**1. Password file.** Create `hosting/.env` (never commit it):

```
WORKSHOP_PASSWORD='the password you will give participants'
WORKSHOP_ADMIN_PASSWORD='a random string of at least 32 characters'
```

**2. Build.** This copies the data and notebook into the workshop folder, then builds the two images (participant
environment and hub):

```bash
cd hosting
./build.sh
```

**3. Start** the hub. It listens only on this machine, at http://127.0.0.1:8799.

```bash
docker compose up -d
```

**4. Publish** it on the internet with a valid HTTPS certificate:

```bash
tailscale funnel --bg --https=8443 http://127.0.0.1:8799
tailscale funnel status          # shows the public address, ending in :8443
```

**Unpublish** with `tailscale funnel --https=8443 off`. Do not use `tailscale funnel reset`: it also removes anything
else this machine publishes.

**Stop everything** with `docker compose down`. Stopping the hub also stops and removes every participant's server;
their folders under `users/` stay until you delete them.

**During the session**, log in with the name `admin` and the admin password, then open `/hub/admin` to see who is
connected and to stop or restart anyone's server.

## What each participant gets

| | |
|---|---|
| Processor | up to 4 cores |
| Memory | up to 6 GB |
| Files | their own copy of the notebook (kept between logins) plus the shared read-only `data` folder |
| Network | none: no internet, and no access to anything else on the host machine |
| Terminal | disabled |

Twelve people working at once use at most 48 cores and 72 GB of memory. The data is covered by the AI-READI Synthetic
Data License Agreement (see `data/README.md`); share the address and password only with people the license covers.
