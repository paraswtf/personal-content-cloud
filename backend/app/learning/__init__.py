"""Learning from the owner's choices (docs/ARCHITECTURE.md section 18).

Modules: events (record corrections), doc_type, images, misclick, people (one learner each), train
(train all, schedule, status, reset), store (learned files). The functions below are the contract the
pipeline and the API call; every learner answers None (nothing learned) until it is trained.
"""
