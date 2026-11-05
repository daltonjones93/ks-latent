# Sourced by the kit's job scripts: make the `module` command available in
# non-interactive shells and load the modules recorded in kit/cashew.conf.
KIT_DIR=${KIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
if ! type module >/dev/null 2>&1; then
  for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash \
           /usr/share/Modules/init/bash; do
    [ -f "$f" ] && . "$f" && break
  done
fi
MODULES=""; NONETCDF=0
[ -f "$KIT_DIR/cashew.conf" ] && . "$KIT_DIR/cashew.conf"
if [ -n "$MODULES" ]; then module load $MODULES; fi
export NONETCDF
