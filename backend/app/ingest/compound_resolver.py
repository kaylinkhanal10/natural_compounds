import logging

logger = logging.getLogger(__name__)

class CompoundResolver:
    """
    Resolves compound identity based on a strict hierarchy:
    1. INCHIKEY (Exact Match) - Primary
    2. CID (If available)
    3. SMILES (Canonical string match)
    """

    def resolve(self, row):
        """
        Resolves the identity of a compound from a data row.
        
        Args:
            row (dict): A dictionary or pandas Series containing compound data.
                        Expected keys (normalized): 'inchikey', 'cid', 'smiles', 'source_id'

        Returns:
            tuple: (canonical_id, method, confidence)
                - canonical_id (str): The stable identifier for the compound.
                - method (str): The method used ("INCHIKEY", "CID", "SMILES", "NEW").
                - confidence (str): "HIGH", "MEDIUM", "LOW".
        """
        
        # 1. Check InChIKey (Primary)
        inchikey = str(row.get('inchikey', '')).strip()
        if inchikey and inchikey.lower() != 'nan' and inchikey.lower() != 'none':
            return inchikey, "INCHIKEY", "HIGH"

        # 2. Check CID
        cid = str(row.get('cid', '')).strip()
        if cid and cid.lower() != 'nan' and cid.lower() != 'none':
            # CID defaults to MEDIUM confidence if InChIKey is missing
            return f"CID:{cid}", "CID", "MEDIUM"

        # 3. Check SMILES
        smiles = str(row.get('smiles', '')).strip()
        if smiles and smiles.lower() != 'nan' and smiles.lower() != 'none':
            # SMILES match is lower confidence as strings can vary (though we can assume canonical input)
            return f"SMILES:{hash(smiles)}", "SMILES", "MEDIUM"

        # 4. Fallback -> Create new with LOW confidence using source_id
        source_id = str(row.get('source_id', '')).strip()
        if source_id:
            return f"source:{source_id}", "SOURCE_ID", "LOW"
            
        return None, "UNKNOWN", "LOW"
