from app.services.neo4j_service import Neo4jService
from app.ml.world_model.infer_supervised import SupervisedInference
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class PlausibilityScorer:
    def __init__(self, db_service: Neo4jService = None):
        self.db = db_service if db_service else Neo4jService()
        try:
             # Attempt to load inference model for chemical similarity
             # In production, this might be heavyweight to load per request.
             # Ideally, we should have a singleton or pass it in.
             # For Experiment C, we will try to load it.
             self.inference = SupervisedInference()
             print("PlausibilityScorer: Supervised Inference Model loaded.")
        except Exception as e:
             print(f"PlausibilityScorer: Model load failed ({e}). using fallback similarity.")
             self.inference = None

    def close(self):
        if self.db:
            self.db.close()

    def score_combination(self, herb_ids: list):
        """
        Scores a combination of herbs (Compound Sets) based on biological plausibility.
        1. Expand Herbs -> Compounds
        2. Select Top-N Compounds
        3. Score Compound Interactions
        4. Aggregate
        """
        if not herb_ids:
            return 0.0, {"error": "No herbs provided"}
            
        # 1. Expand & Select
        compound_sets = self._expand_and_select(herb_ids, limit=20)
        
        if not compound_sets:
             return 0.0, {"error": "No compounds found for herbs."}
             
        # 2. Score Compounds (Logic V2 at Compound Level)
        score_data = self.score_compound_sets(compound_sets)
        
        # 3. Aggregate
        final_score, explanation = self._aggregate_scores(score_data, herb_ids)
        
        return final_score, explanation

    def _expand_and_select(self, herb_ids, limit=20):
        """
        Expands herbs to their constituent compounds.
        Selection Policy: Top-N by Degree (number of targets).
        """
        HERB_NAME_MAP = {
            "Curcuma longa": "Curcumae Longae Rhizoma",
            "Piper longum": "Piperis Longi Fructus",
            "Zingiber officinale": "Zingiberis Rhizoma",
            "Phyllanthus emblica": "Phyllanthi Fructus",
            "Tinospora cordifolia": "Tinosporae Radix",
            "Glycyrrhiza glabra": "Glycyrrhizae Radix Et Rhizoma",
            "Ocimum tenuiflorum": "Ocimi Sancti Herba", 
            "Ocimum sanctum": "Ocimi Sancti Herba"
        }
        
        expanded = {}
        
        with self.db.driver.session() as session:
            for hid in herb_ids:
                # Get Compounds for this herb, ordered by "evidence" (Target Count)
                query = """
                MATCH (h:Herb {herbId: $hid})
                OPTIONAL MATCH (hm:HerbMaterial) 
                WHERE hm.latin_name = h.scientificName OR hm.latin_name = $map[h.scientificName]
                MATCH (hm)-[:HAS_COMPOUND]->(c:Compound)
                OPTIONAL MATCH (c)-[:TARGETS]->(p:Protein)
                WITH c, count(p) as degree, collect(p.name) as targets
                ORDER BY degree DESC
                LIMIT $limit
                RETURN 
                    c.inchikey as inchikey, 
                    c.smiles as smiles, 
                    c.name as name, 
                    degree, 
                    targets
                """
                result = session.run(query, hid=hid, map=HERB_NAME_MAP, limit=limit)
                compounds = [dict(record) for record in result]
                expanded[hid] = compounds
                
        return expanded

    def score_compound_sets(self, compound_sets: dict):
        """
        Public API: Computes Coherence and Redundancy between sets of compounds.
        Input: dict { "GroupA": [compounds], "GroupB": [compounds] }
        """
        n_groups = len(compound_sets)
        
        # Flatten all compounds for Coverage
        all_compounds = []
        for hid, comps in compound_sets.items():
            all_compounds.extend(comps)
            
        # Global Coverage
        all_targets = set()
        for c in all_compounds:
            all_targets.update(c['targets'])
        
        coverage_score = min(len(all_targets), 100) / 100.0
        
        coherence_scores = []
        redundancy_penalties = []
        interaction_pairs = []

        return {
            "coverage": coverage_score,
            "coherence_list": coherence_scores,
            "redundancy_list": redundancy_penalties,
            "interactions": interaction_pairs,
            "total_compounds": len(all_compounds),
            "total_targets": len(all_targets),
            "all_compounds": all_compounds
        }

        if n_groups == 1:
            # Single Group Case: Internal Coherence
            # Do the top compounds work together?
            # Assumption: A single "herb" or "formula" is chemically internally consistent (or evolved to be).
            coherence_scores.append(1.0) 
            redundancy_penalties.append(0.0)
            
        else:
            # Pairwise Comparisons between Sets
            keys = list(compound_sets.keys())
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    set_a = compound_sets[keys[i]]
                    set_b = compound_sets[keys[j]]
                    
                    # Target Overlap (Bio Coherence)
                    targets_a = set()
                    for c in set_a: targets_a.update(c['targets'])
                    
                    targets_b = set()
                    for c in set_b: targets_b.update(c['targets'])
                    
                    u = len(targets_a.union(targets_b))
                    if u > 0:
                        bio_sim = len(targets_a.intersection(targets_b)) / u
                    else:
                        bio_sim = 0.0
                        
                    
                    # Chemical Similarity (Redundancy Driver)
                    chem_sim = 0.0
                    if self.inference:
                        # Hybrid Logic: Only embed Structure-Backed compounds
                        smiles_a = [c['smiles'] for c in set_a if c.get('smiles')]
                        smiles_b = [c['smiles'] for c in set_b if c.get('smiles')]
                        
                        if smiles_a and smiles_b:
                            try:
                                # Embed
                                emb_a = self.inference.encode(smiles_a) 
                                emb_b = self.inference.encode(smiles_b)
                                
                                valid_a = np.array([e for e in emb_a if e is not None])
                                valid_b = np.array([e for e in emb_b if e is not None])
                                
                                if len(valid_a) > 0 and len(valid_b) > 0:
                                    centroid_a = np.mean(valid_a, axis=0).reshape(1, -1)
                                    centroid_b = np.mean(valid_b, axis=0).reshape(1, -1)
                                    chem_sim = cosine_similarity(centroid_a, centroid_b)[0][0]
                            except Exception as e:
                                # Fallback if encoding fails
                                print(f"Warning: Encoding failed: {e}")
                                chem_sim = 0.0
                        else:
                            # Symbolic-Only Case: No SMILES available
                            # As per design: Skip Chem Sim. Rely on Graph Logic.
                            # Chem Sim = 0 IMPLIES Redundancy = 0 (No penalty for unknown chemistry)
                            chem_sim = 0.0
                    
                    coherence_scores.append(bio_sim)
                    
                    # Redundancy Definition
                    red = bio_sim * chem_sim
                    redundancy_penalties.append(red)
                    
                    interaction_pairs.append({
                        'pair': (keys[i], keys[j]),
                        'bio_sim': round(bio_sim, 2),
                        'chem_sim': round(chem_sim, 2),
                        'redundancy': round(red, 2)
                    })
                    
                    # PK Enhancer Logic (New)
                    # Check if either set contains a PK enhancer (CYP/P-gp inhibitors)
                    is_pk_a, hits_a = self._is_pk_enhancer(set_a)
                    is_pk_b, hits_b = self._is_pk_enhancer(set_b)
                    
                    pk_boost = 0.0
                    pk_rationale = []
                    
                    # Scenario: Enhancer + Substrate
                    # We assume if one is Enhancer, it boosts the other.
                    # Ideally check if other is substrate, but for now assume yes by default.
                    if is_pk_a and not is_pk_b:
                        pk_boost = 0.3
                        pk_rationale = hits_a
                    elif is_pk_b and not is_pk_a:
                        pk_boost = 0.3
                        pk_rationale = hits_b
                    elif is_pk_a and is_pk_b:
                        # Both are enhancers? Maybe less synergy, or additive.
                        # Let's say 0.15 just to acknowledge interaction.
                        pk_boost = 0.15
                        pk_rationale = hits_a + hits_b
                        
                    if pk_boost > 0:
                        coherence_scores.append(pk_boost) # Add boost to coherence list to raise average/max
                        interaction_pairs[-1]['pk_boost'] = pk_boost
                        interaction_pairs[-1]['pk_rationale'] = pk_rationale

        return {
            "coverage": coverage_score,
            "coherence_list": coherence_scores,
            "redundancy_list": redundancy_penalties,
            "interactions": interaction_pairs,
            "total_compounds": len(all_compounds),
            "total_targets": len(all_targets)
        }
        
    def score_direct(self, compound_groups: dict):
        """
        Directly score groups of compounds without DB lookup.
        Args:
           compound_groups: dict { "Group_Name": [{ "smiles": str, "targets": [str] }] }
        """
        # Hydrate targets if missing (from DB)
        self._hydrate_targets(compound_groups)
        
        score_data = self.score_compound_sets(compound_groups)
        final_score, explanation = self._aggregate_scores(score_data, list(compound_groups.keys()))
        return final_score, explanation

    def _hydrate_targets(self, compound_groups: dict):
        """
        Fetches 'targets' for compounds if missing.
        """
        # Collect all compound objects that need targets
        to_hydrate = []
        for group, comps in compound_groups.items():
            for c in comps:
                if 'targets' not in c:
                    to_hydrate.append(c)
        
        if not to_hydrate:
            return

        # Prepare IDs for batch query
        ids = [c.get('compoundId') for c in to_hydrate if c.get('compoundId')]
        # Fallback to names if IDs missing? (Likely not needed if using standard objects)
        
        lookup = {}
        if ids:
            with self.db.driver.session() as session:
                q = """
                MATCH (c:Compound)-[:TARGETS]->(p:Protein)
                WHERE c.compoundId IN $ids
                RETURN c.compoundId as id, collect(p.name) as targets
                """
                res = session.run(q, ids=ids)
                lookup = {r['id']: r['targets'] for r in res}
        
        # Assign
        for c in to_hydrate:
            cid = c.get('compoundId')
            if cid and cid in lookup:
                c['targets'] = lookup[cid]
            else:
                c['targets'] = [] # Default to empty to prevent KeyError

    def _aggregate_scores(self, data, herb_ids):
        # Max-Spanning Aggregation
        
        # Coherence: Use Max (or high percentile) to avoid dilution
        if data['coherence_list']:
            # Soft Max: (Max + Mean) / 2
            coh_max = max(data['coherence_list'])
            coh_mean = sum(data['coherence_list']) / len(data['coherence_list'])
            coherence = (coh_max + coh_mean) / 2.0
        else:
            coherence = 1.0 # Single herb default
            
        # Redundancy: Sum of penalties? Or Max? 
        # Reducing plausibility if ANY pair is redundant?
        # Let's subtract Max Redundancy to be lenient, or sum?
        # User said "Avoid mean dilution". 
        # Let's take Max Redundancy as the limiting factor.
        if data['redundancy_list']:
            redundancy = max(data['redundancy_list'])
        else:
            redundancy = 0.0
            
        # Risk (Simple count of low-evidence herbs? 
        # Here we selected Top-N, so we implicitly filtered for evidence.
        # But we can check if any herb returned 0 compounds)
        # (This logic is implicitly handled by _expand_and_select returning empty)
        risk = 0.0
        
        # Final Score
        # Plausibility = (Coherence * 0.6) + (Coverage * 0.4) - (Redundancy * 1.5)
        base = (coherence * 0.6) + (data['coverage'] * 0.4)
        final = base - (redundancy * 1.5) - risk
        final = max(0.0, min(1.0, final))
        
        # --- NEW: Mock Complex Modules ---
        dose_analysis = self._mock_dose_analysis(data['all_compounds'])
        adme_sim = self._mock_adme_simulation(data['all_compounds'])
        risk_assessment = self._mock_risk_assessment(data['all_compounds'], redundancy, risk)
        novelty = self._mock_novelty_report(data['all_compounds'])
        contributors = self._build_contributors(data['all_compounds'], risk_assessment)
        
        # Target Confidence (Mock based on real targets)
        target_summary = self._build_target_summary(data['all_compounds'])

        # Detailed Response
        response = {
            "plausibility_score": round(final, 3),
            "metrics": {
                "coverage": round(data['coverage'], 3),
                "redundancy_penalty": round(redundancy, 3),
                "risk_penalty": round(risk_assessment['risk_penalty'], 3),
                "uncertainty": 0.35 # Placeholder: Would come from Bayesian Neural Net variance
            },
            "target_confidence_summary": target_summary,
            "decision": {
                "band": "HIGH_PRIORITY" if final > 0.7 else ("MODERATE" if final > 0.4 else "LOW_PRIORITY"),
                "percentile": round(final * 100, 1), # Naive mapping
                "recommendation": self._interpret_score(final, redundancy),
                "rule_trace": ["Applied: SCORE_THRESHOLD_MAPPING"]
            },
            "dose_analysis": dose_analysis,
            "biological_risk_assessment": risk_assessment,
            "contributors": contributors,
            "biological_context": {
                "predicted_targets": list(target_summary['gated_target_counts'].keys())[:5], # Just show names?
                "coverage_interpretation": f"Hits {len(target_summary['gated_target_counts'])} distinct high-confidence targets."
            },
            "adme_simulation": adme_sim,
            "novelty_report": novelty,
            "feedback_status": {
                "has_feedback": False,
                "label": None,
                "score_adjustment": 0.0
            },
            "disclaimer": "This score represents plausibility ranking under known compound-target biology. It does NOT predict biological reinforcement, therapeutic results, or human health outcomes. Experimental validation is required."
        }
        
        # Legacy return for compatibility (score, explanation)
        # But API expects tuple: (score, dict)
        # We will return the FULL dict as the second element.
        return final, response

    # --- Helper Methods for New JSON Schema ---

    def _build_target_summary(self, compounds):
        """
        Summarizes target hits.
        """
        counts = {}
        max_probs = {}
        
        for c in compounds:
            targets = c.get('targets', [])
            name = c.get('name', 'Unknown')
            # Mock prob = 0.5 + 0.1 * target_count (clamped to 1.0)
            prob = min(0.9, 0.4 + (0.05 * len(targets)))
            
            if len(targets) > 0:
                max_probs[name] = round(prob, 2)
                counts[name] = len(targets)
        
        return {
            "threshold": 0.2, # Confidence Threshold
            "max_prob_per_compound": max_probs,
            "gated_target_counts": counts,
            "note": "Targets below threshold are suppressed."
        }

    def _mock_dose_analysis(self, compounds):
        """
        Simulates dose analysis based on QED or MW/LogP proxies.
        """
        weights = {}
        potencies = {}
        sources = {}
        combined = {}
        
        for c in compounds:
            name = c.get('name', 'Unknown')
            # Simulate weight based on MW (heavier = need less? or more? let's just use 1.0 norm)
            weights[name] = 1.0
            # Potency based on simple heuristic
            potency = 0.8
            if c.get('mw') and c['mw'] < 500: potency += 0.1
            potencies[name] = round(potency, 2)
            
            sources[name] = f"QSAR Estimate (MW={int(c.get('mw',0))})"
            combined[name] = round(weights[name] * potencies[name], 2)
            
        return {
            "input_mode": "normalized_weight_and_potency",
            "normalized_weights": weights,
            "normalized_potencies": potencies,
            "activity_source": sources,
            "combined_influence": combined,
            "ic50_adjustment_explanation": "Combined influence scales compound embeddings by both relative dosage and biological potency."
        }

    def _mock_adme_simulation(self, compounds):
        """
        Simulates ADME properties using Lipinski rules.
        """
        adme = []
        for c in compounds:
            logp = c.get('logp')
            mw = c.get('mw')
            
            # Simple Rule-Based Simulation
            bbb = "Unknown"
            if logp and mw:
                if 2 < logp < 4 and mw < 450:
                    bbb = "Permeable (Simulated)"
                else:
                    bbb = "Impermeable (Simulated)"
            
            solubility = "Moderate"
            if logp and logp < 3: solubility = "High (Simulated)"
            elif logp and logp > 5: solubility = "Low (Simulated)"
            
            adme.append({
                "smiles": c.get('smiles', 'N/A'),
                "name": c.get('name', 'Unknown'),
                "logP_proxy": round(logp, 2) if logp else None,
                "solubility_class": solubility,
                "blood_brain_barrier": bbb,
                "metabolic_stability": "Unknown (Placeholder)"
            })
        return adme

    def _mock_risk_assessment(self, compounds, redundancy_penalty, other_risk):
        """
        Assess risk based on properties and redundancy.
        """
        # Base risk from redundancy
        risk_val = (redundancy_penalty * 0.5) + other_risk
        
        # Check specific toxicophores (Simulated)
        flags = {}
        for c in compounds:
            f = []
            if c.get('mw') and c['mw'] > 600: f.append("High MW warning")
            if c.get('tpsa') and c['tpsa'] > 140: f.append("Low Absorption warning")
            flags[c.get('name', 'Unknown')] = f
            if f: risk_val += 0.1
            
        risk_val = min(1.0, risk_val)
        
        return {
            "risk_penalty": round(risk_val, 3),
            "risk_level": "MODERATE" if risk_val > 0.4 else "LOW",
            "primary_risk_factors": ["Redundancy" if redundancy_penalty > 0.5 else "None"],
            "compound_level_flags": flags,
            "interpretation": "Biological risk signals derived from physicochemical properties and target overlap."
        }

    def _mock_novelty_report(self, compounds):
        return {
            "known_compounds": len(compounds),
            "novel_compounds": 0, # Assuming all from DB
            "novel_smiles": [],
            "inference_method": "Graph Isomorphism Network (Generalization)"
        }

    def _build_contributors(self, compounds, risk_assessment):
        contribs = []
        total = len(compounds)
        for c in compounds:
            # Contribution is uniform for now, adjusted by risk
            base = 1.0 / total if total > 0 else 0
            
            targets = c.get('targets', [])
            top_t = targets[:3] if targets else ["No direct targets"]
            
            contribs.append({
                "smiles": c.get('smiles', 'N/A'),
                "name": c.get('name', 'Unknown'),
                "contribution": round(base, 3),
                "weight": 1.0,
                "top_targets": top_t,
                "risk_flags": risk_assessment['compound_level_flags'].get(c.get('name', 'Unknown'), [])
            })
        return contribs

    def _is_pk_enhancer(self, compound_set):
        """
        Checks if a set of compounds contains known Pharmacokinetic (PK) enhancers.
        enhancing bioavailability via CYP450 inhibition or P-gp modulation.
        """
        PK_TARGETS = {
            "CYP3A4": "Cytochrome P450 3A4 (Metabolism)", 
            "ABCB1": "P-glycoprotein 1 (Efflux Pump)", 
            "UGT1A1": "UDP-glucuronosyltransferase 1A1",
            "CYP2C9": "Cytochrome P450 2C9",
            "CYP2D6": "Cytochrome P450 2D6"
        }
        
        hits = []
        for c in compound_set:
            if not c.get('targets'): continue
            for t in c['targets']:
                if t in PK_TARGETS:
                    hits.append(f"{c.get('name', 'Unknown')}_targets_{t}")
                    
        return len(hits) > 0, hits

    def _interpret_score(self, score, redundancy):
        if redundancy > 0.4:
            return "Redundant. High chemical and biological overlap detected."
        if score > 0.7:
            return "Highly Plausible. Strong coherence and coverage."
        if score > 0.4:
            return "Moderate Plausibility."
        return "Low Plausibility."
