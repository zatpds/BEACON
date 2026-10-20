"""
StackWood -- robosuite Stack with a wood table texture.

Identical physics to Stack; only the table visual material is swapped from
ceramic to wood-varnished-panels via the edit_model_xml hook.  This avoids
touching the shared table_arena.xml and lets both Stack (ceramic) and
StackWood coexist in the same process.

Registered automatically by robosuite's EnvMeta metaclass, so you can do:
    robosuite.make("StackWood", ...)
after importing this module.
"""

import xml.etree.ElementTree as ET

from robosuite.environments.manipulation.stack import Stack


WOOD_TEXTURE_FILE = "wood-varnished-panels.png"


class StackWood(Stack):
    """Stack task with a wood table surface instead of ceramic."""

    def edit_model_xml(self, xml_str):
        xml_str = super().edit_model_xml(xml_str)

        root = ET.fromstring(xml_str)
        asset = root.find("asset")

        for tex in asset.findall("texture"):
            if tex.get("name") != "tex-ceramic":
                continue
            fpath = tex.get("file", "")
            if fpath.endswith("ceramic.png"):
                tex.set("file", fpath.replace("ceramic.png", WOOD_TEXTURE_FILE))
            break

        return ET.tostring(root, encoding="utf8").decode("utf8")
