import os
import shutil

from dockerspawner.volumenamingstrategy import escaped_format_volume_name

c = get_config()  # noqa: F821

# Workshop files live on a fast drive: the shared read-only data, the notebook template, and one folder per person.
STORAGE = os.environ.get("WORKSHOP_STORAGE", "/mnt/fast/aireadi-workshop")
USER_HOME = f"{STORAGE}/users/{{username}}"

# Log in with any name plus the shared workshop password; each name gets its own server.
# The name "admin" with the admin password opens the admin page (stop or restart anyone's server).
c.JupyterHub.authenticator_class = "shared-password"
c.SharedPasswordAuthenticator.user_password = os.environ["WORKSHOP_PASSWORD"]
c.SharedPasswordAuthenticator.admin_password = os.environ["WORKSHOP_ADMIN_PASSWORD"]
c.Authenticator.admin_users = {"admin"}
c.Authenticator.allow_all = True
c.JupyterHub.template_vars = {
    "announcement_login": "AI-READI workshop: type your first name and the workshop password. "
                          "Your own copy of the notebook opens in about 20 seconds.",
}

# One container per participant, started from the workshop image and thrown away when stopped.
# Their folder on the fast drive stays, so work survives a restart.
c.JupyterHub.spawner_class = "docker"
c.DockerSpawner.image = os.environ.get("WORKSHOP_IMAGE", "aireadi-workshop-user:latest")
c.DockerSpawner.network_name = "aireadi_internal"      # no internet, no access to this machine's other services
c.DockerSpawner.use_internal_ip = True
c.DockerSpawner.remove = True
c.DockerSpawner.notebook_dir = "/home/jovyan"
c.DockerSpawner.format_volume_name = escaped_format_volume_name
c.DockerSpawner.volumes = {
    USER_HOME: "/home/jovyan",
    f"{STORAGE}/data": {"bind": "/home/jovyan/data", "mode": "ro"},
}


def prepare_home(spawner):
    # First login: create the person's folder and give them a fresh copy of the notebook.
    home = escaped_format_volume_name(USER_HOME, spawner)
    notebook = os.path.join(home, "aireadi_synthetic_demo.ipynb")
    os.makedirs(os.path.join(home, "data"), exist_ok=True)      # mount point for the shared data
    if not os.path.exists(notebook):
        shutil.copy(os.path.join(STORAGE, "template", "aireadi_synthetic_demo.ipynb"), notebook)
    for path in (home, notebook):
        os.chown(path, 1000, 100)                                # the participant's user inside the container


c.Spawner.pre_spawn_hook = prepare_home
c.DockerSpawner.extra_host_config = {"pids_limit": 1024}
c.Spawner.default_url = "/lab/tree/aireadi_synthetic_demo.ipynb"
c.Spawner.args = ["--ServerApp.terminals_enabled=False"]
c.Spawner.cpu_limit = 4
c.Spawner.mem_limit = "6G"
# match the math libraries' thread count to the 4-core limit, so 12 people at once do not oversubscribe the machine
c.Spawner.environment = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
c.Spawner.start_timeout = 180
c.Spawner.http_timeout = 120
c.JupyterHub.active_server_limit = 20

# The hub runs in a container; participant containers reach it by name on the internal network.
c.JupyterHub.bind_url = "http://0.0.0.0:8000"
c.JupyterHub.hub_ip = "0.0.0.0"
c.JupyterHub.hub_connect_ip = "aireadi-hub"

# Keep logins and the database across hub restarts.
c.JupyterHub.db_url = "sqlite:////srv/jupyterhub/state/jupyterhub.sqlite"
c.JupyterHub.cookie_secret_file = "/srv/jupyterhub/state/jupyterhub_cookie_secret"

# Stop servers that have been idle for two hours.
c.JupyterHub.load_roles = [{
    "name": "idle-culler",
    "scopes": ["list:users", "read:users:activity", "read:servers", "delete:servers"],
    "services": ["idle-culler"],
}]
c.JupyterHub.services = [{
    "name": "idle-culler",
    "command": ["python3", "-m", "jupyterhub_idle_culler", "--timeout=7200"],
}]
