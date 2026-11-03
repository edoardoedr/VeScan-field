"""vescan_field: field-based pipeline from the haversack ranked store (no VMTK).

Generic part:
    config        typed JSON config (batch + pipeline, one section per stage)
    log           logging: console + one file per patient
    store         store reader (zarr)
    field         CT grid geometry and a structure's margin from the store
    io            JSON, images, markups
    surface       VTK surface utilities shared by the stages
    group_tree    the group graph of the topology JSON
    centerline_distance  nearest centerline segment per point (vtkvmtkPolyBallLine's metric)
    lobes         lobe order, mask names, sides
    orchestrator  patient -> structures -> field -> enabled stages

Stages (vescan_field/stages/<name>/, see vescan_field/stages/__init__.py).
"""
