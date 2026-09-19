"""Variantes controladas A0--A6 para estudos de ablação pareados."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping
import yaml
from jsonschema import Draft202012Validator
from cinebot_ml.config import PROJECT_ROOT

DEFAULT_ABLATION_CONFIG_PATH=PROJECT_ROOT/"configs/experiments/ablation_v1.yaml"
DEFAULT_ABLATION_SCHEMA_PATH=PROJECT_ROOT/"configs/experiments/ablation_v1.schema.json"
class AblationValidationError(ValueError): pass

@dataclass(frozen=True)
class AblationContext:
    agent_id:str; seed:int; candidate_set_id:str
    history:tuple[Mapping[str,Any],...]; profile:Mapping[str,Any]; item_features:Mapping[str,Any]

@dataclass(frozen=True)
class AblationVariant:
    name:str; version:str; removed:tuple[str,...]; active:tuple[str,...]
    @property
    def variant_id(self):
        return hashlib.sha256(json.dumps({"name":self.name,"version":self.version,"removed":self.removed,"active":self.active},sort_keys=True).encode()).hexdigest()[:24]

@dataclass(frozen=True)
class AblationReport:
    study_id:str; manifest:Mapping[str,Any]; rows:tuple[Mapping[str,Any],...]

def load_ablation_variants(path:Path=DEFAULT_ABLATION_CONFIG_PATH):
    payload=yaml.safe_load(path.read_text()); schema=json.loads(DEFAULT_ABLATION_SCHEMA_PATH.read_text())
    error=next(iter(Draft202012Validator(schema).iter_errors(payload)),None)
    if error: raise AblationValidationError(error.message)
    digest=hashlib.sha256(path.read_bytes()).hexdigest(); lock=json.loads(path.with_suffix(".lock.json").read_text())
    if lock.get("sha256")!=digest: raise AblationValidationError("Configuração de ablação alterada sem lock.")
    expected={"A0":(),"A1":("history",),"A2":("negative_feedback",),"A3":("genres",),"A4":("director",),"A5":("decade_popularity",),"A6":("text",)}
    components=tuple(payload["components"]); variants=[]
    for name,spec in payload["variants"].items():
        removed=tuple(spec["removed"]); active=tuple(item for item in components if item not in removed)
        if removed!=expected[name]: raise AblationValidationError(f"{name} remove componentes diferentes do protocolo.")
        variants.append(AblationVariant(name,str(payload["version"]),removed,active))
    return tuple(variants),digest

def apply_ablation(context:AblationContext, variant:AblationVariant)->AblationContext:
    history=tuple(dict(item) for item in context.history); profile=dict(context.profile); features=dict(context.item_features)
    for component in variant.removed:
        if component=="history": history=()
        elif component=="negative_feedback": history=tuple(item for item in history if item.get("feedback")!="dislike")
        elif component=="genres": profile.pop("ranked_genres",None); features.pop("genres",None)
        elif component=="director": profile.pop("preferred_directors",None); features.pop("director",None)
        elif component=="decade_popularity":
            for key in ("decade_preference","popularity_preference"): profile.pop(key,None)
            for key in ("decade","popularity"): features.pop(key,None)
        elif component=="text":
            for key in ("preferred_keywords",): profile.pop(key,None)
            for key in ("synopsis","keywords","text"): features.pop(key,None)
        else: raise AblationValidationError(f"Componente desconhecido: {component}")
    return replace(context,history=history,profile=profile,item_features=features)

def execute_ablation(context:AblationContext, runner:Callable[[AblationVariant,AblationContext],Mapping[str,Any]], path:Path=DEFAULT_ABLATION_CONFIG_PATH)->AblationReport:
    variants,digest=load_ablation_variants(path); rows=[]
    for variant in variants:
        transformed=apply_ablation(context,variant); result=dict(runner(variant,transformed))
        if "ndcg_at_k" not in result: raise AblationValidationError("Runner deve registrar ndcg_at_k.")
        rows.append({"variant":variant.name,"variant_id":variant.variant_id,"agent_id":context.agent_id,"seed":context.seed,"candidate_set_id":context.candidate_set_id,"active_components":list(variant.active),"removed_components":list(variant.removed),**result})
    identity={"config_sha256":digest,"agent_id":context.agent_id,"seed":context.seed,"candidate_set_id":context.candidate_set_id,"variants":[row["variant_id"] for row in rows]}
    study_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    return AblationReport(study_id,{**identity,"study_id":study_id},tuple(rows))
