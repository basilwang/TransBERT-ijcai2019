from __future__ import absolute_import

import lucene


global _vm
_vm = None


def get_vm():
    global _vm
    if _vm is None:
        _vm = lucene.initVM(vmargs="-Djava.awt.headless=true")
    return _vm
