#!/usr/bin/env python3

import sys
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
import multiprocessing as mp
from multiprocessing import Pool
import time
import pandas as pd
import os
import json
from docopt import docopt
import pkg_resources
import csv

cmd_str = """Usage:
rd_filters filter --in INPUT_FILE --prefix PREFIX [--rules RULES_FILE_NAME] [--alerts ALERT_FILE_NAME][--np NUM_CORES]
rd_filters template --out TEMPLATE_FILE [--rules RULES_FILE_NAME]

Options:
--in INPUT_FILE input file name
--prefix PREFIX prefix for output file names
--rules RULES_FILE_NAME name of the rules JSON file
--alerts ALERTS_FILE_NAME name of the structural alerts file
--np NUM_CORES the number of cpu cores to use (default is all)
--out TEMPLATE_FILE parameter template file name
"""


def read_rules(rules_file_name):
    """
    Read rules from a JSON file
    :param rules_file_name: JSON file name
    :return: dictionary corresponding to the contents of the JSON file
    """
    with open(rules_file_name) as json_file:
        try:
            rules_dict = json.load(json_file)
            return rules_dict
        except json.JSONDecodeError:
            print(f"Error parsing JSON file {rules_file_name}")
            sys.exit(1)


def write_rules(rule_dict, file_name):
    """
    Write configuration to a JSON file
    :param rule_dict: dictionary with rules
    :param file_name: JSON file name
    :return: None
    """
    ofs = open(file_name, "w")
    ofs.write(json.dumps(rule_dict, indent=4, sort_keys=True))
    print(f"Wrote rules to {file_name}")
    ofs.close()


def default_rule_template(alert_list, file_name):
    """
    Build a default rules template
    :param alert_list: list of alert set names
    :param file_name: output file name
    :return: None
    """
    default_rule_dict = {
        "MW": [0, 500],
        "LogP": [-5, 5],
        "HBD": [0, 5],
        "HBA": [0, 10],
        "TPSA": [0, 200],
        "Rot": [0, 10]
    }
    for rule_name in alert_list:
        if rule_name == "Inpharmatica":
            default_rule_dict["Rule_" + rule_name] = True
        else:
            default_rule_dict["Rule_" + rule_name] = False
    write_rules(default_rule_dict, file_name)


def get_config_file(file_name, environment_variable):
    """
    Read a configuration file, first look for the file, if you can't find
    it there, look in the directory pointed to by environment_variable
    :param file_name: the configuration file
    :param environment_variable: the environment variable
    :return: the file name or file_path if it exists otherwise exit
    """
    if os.path.exists(file_name):
        return file_name
    else:
        config_dir = os.environ.get(environment_variable)
        if config_dir:
            config_file_path = os.path.join(os.path.sep, config_dir, file_name)
            if os.path.exists(config_file_path):
                return config_file_path

    error_list = [f"Could not file {file_name}"]
    if config_dir:
        err_str = f"Could not find {config_file_path} based on the {environment_variable}" + \
                  "environment variable"
        error_list.append(err_str)
    error_list.append(f"Please check {file_name} exists")
    error_list.append(f"Or in the directory pointed to by the {environment_variable} environment variable")
    print("\n".join(error_list))
    sys.exit(1)


class RDFilters:
    def __init__(self, params, verbose=False):
        """Initialize RDFilters object and setup filtering rules
        :param params: dict of filter parameters
        :param verbose: boolean for reporting
        """
        self.params = params
        self.verbose = verbose
        self.rule_list = []  # [(pattern, max_count, description), ...]
        self.rule_dict = {}  # {rule_name: (min, max), ...}
        self.alert_names = []  # Store alert names
        self.alert_priorities = {}  # Store priorities for each alert
        self.alert_smarts = {}  # Store SMARTS patterns for each alert
        
        # Read in alert data
        alert_filename = params.get("alert_filename", "")
        if alert_filename:
            with open(alert_filename, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    rule_id, rule_set, desc, smarts, rule_set_name, priority, max_val = row
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern is None:
                        if verbose:
                            print(f"Invalid SMARTS pattern: {smarts}", file=sys.stderr)
                        continue
                    self.rule_list.append((pattern, int(max_val), desc))
                    self.alert_names.append(desc)
                    self.alert_priorities[desc] = int(priority)
                    self.alert_smarts[desc] = smarts

    def evaluate(self, lst_in):
        """
        Evaluate structure alerts on a list of SMILES
        :param lst_in: input list of [SMILES, Name]
        :return: list of alerts matched with properties and alert columns with matching substructures
        """
        smiles, name = lst_in
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Create a list of empty strings for each alert
            alert_results = [""] * len(self.alert_names)
            alert_smarts_results = [""] * len(self.alert_names)
            return [smiles, name, 'INVALID', -999, -999, -999, -999, -999, -999] + alert_results + alert_smarts_results
        
        desc_list = [MolWt(mol), MolLogP(mol), NumHDonors(mol), NumHAcceptors(mol), TPSA(mol),
                     CalcNumRotatableBonds(mol)]
        
        # Initialize alert results
        alert_results = [""] * len(self.alert_names)  # For substructure matches
        alert_smarts_results = [""] * len(self.alert_names)  # For SMARTS patterns
        alerts = []
        
        # Check each alert
        for i, (patt, max_val, desc) in enumerate(self.rule_list):
            matches = mol.GetSubstructMatches(patt)
            if len(matches) > max_val:
                alerts.append(desc)
                # Get the matching substructure SMILES
                match_atoms = set()
                for match in matches:
                    match_atoms.update(match)
                
                # Create a molecule from the matching atoms
                match_mol = Chem.RWMol(mol)
                atoms_to_delete = []
                for atom_idx in range(match_mol.GetNumAtoms()):
                    if atom_idx not in match_atoms:
                        atoms_to_delete.append(atom_idx)
                
                # Delete atoms in reverse order to maintain correct indices
                for atom_idx in sorted(atoms_to_delete, reverse=True):
                    match_mol.RemoveAtom(atom_idx)
                
                # Store both the matching substructure and the SMARTS pattern
                alert_results[i] = Chem.MolToSmiles(match_mol)
                alert_smarts_results[i] = self.alert_smarts[desc]
        
        alert_str = "; ".join(alerts) if alerts else "OK"
        return [smiles, name, alert_str] + desc_list + alert_results + alert_smarts_results


def main():
    cmd_input = docopt(cmd_str)
    alert_file_name = cmd_input.get("--alerts") or pkg_resources.resource_filename('rd_filters',
                                                                                   "data/alert_collection.csv")
    rf = RDFilters(alert_file_name)

    if cmd_input.get("template"):
        template_output_file = cmd_input.get("--out")
        default_rule_template(rf.get_alert_sets(), template_output_file)

    elif cmd_input.get("filter"):
        input_file_name = cmd_input.get("--in")
        rules_file_name = cmd_input.get("--rules") or pkg_resources.resource_filename('rd_filters', "data/rules.json")
        rules_file_path = get_config_file(rules_file_name, "FILTER_RULES_DATA")
        prefix_name = cmd_input.get("--prefix")
        num_cores = cmd_input.get("--np") or mp.cpu_count()
        num_cores = int(num_cores)

        print("using %d cores" % num_cores, file=sys.stderr)
        start_time = time.time()
        p = Pool(num_cores)
        input_data = [x.split() for x in open(input_file_name)]
        input_data = [x for x in input_data if len(x) == 2]
        rule_dict = read_rules(rules_file_path)

        rule_list = [x.replace("Rule_", "") for x in rule_dict.keys() if x.startswith("Rule") and rule_dict[x]]
        rule_str = " and ".join(rule_list)
        print(f"Using alerts from {rule_str}", file=sys.stderr)
        rf.build_rule_list(rule_list)
        res = list(p.map(rf.evaluate, input_data))
        
        # Create column names including alert names
        base_columns = ["SMILES", "NAME", "FILTER", "MW", "LogP", "HBD", "HBA", "TPSA", "Rot"]
        alert_columns = []
        for name in rf.alert_names:
            alert_columns.extend([
                f"{name} (p{rf.alert_priorities[name]}) - substructure",
                f"{name} (p{rf.alert_priorities[name]}) - SMARTS"
            ])
        all_columns = base_columns + alert_columns
        
        df = pd.DataFrame(res, columns=all_columns)
        
        # Create property filter column
        df['PROP_FILTER'] = 'PASS'
        mask = (
            ~df.MW.between(*rule_dict["MW"]) |
            ~df.LogP.between(*rule_dict["LogP"]) |
            ~df.HBD.between(*rule_dict["HBD"]) |
            ~df.HBA.between(*rule_dict["HBA"]) |
            ~df.TPSA.between(*rule_dict["TPSA"]) |
            ~df.Rot.between(*rule_dict["Rot"])
        )
        df.loc[mask, 'PROP_FILTER'] = 'FAIL'

        # Write output files
        output_smiles_file = prefix_name + ".smi"
        output_csv_file = prefix_name + ".csv"
        df[["SMILES", "NAME"]].to_csv(f"{output_smiles_file}", sep=" ", index=False, header=False)
        df.to_csv(f"{output_csv_file}", index=False)
        
        print(f"Wrote all SMILES to {output_smiles_file}", file=sys.stderr)
        print(f"Wrote detailed data to {output_csv_file}", file=sys.stderr)

        # Count statistics
        num_total = df.shape[0]
        num_passed = ((df.FILTER == "OK") & (df.PROP_FILTER == "PASS")).sum()
        fraction_passed = "%.1f" % (num_passed / num_total * 100.0)
        print(f"{num_passed} of {num_total} passed all filters {fraction_passed}%", file=sys.stderr)
        
        elapsed_time = "%.2f" % (time.time() - start_time)
        print(f"Elapsed time {elapsed_time} seconds", file=sys.stderr)


if __name__ == "__main__":
    main()
