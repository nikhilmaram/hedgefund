"""Contains all the functions corresponding to an employee"""

import pandas as pd
from datetime import datetime
from typing import List
from typing import Tuple
import numpy as np
import queue

import config as cfg



class Employee:
    def __init__(self):
        self.immediate_subordinates = []
        self.top_user_in_hierarchy  = ""

    #############################################################
    ## Setter Functions of the variables
    #############################################################

    def set_first_name(self,first_name):
        self.first_name = first_name

    def set_last_name(self,last_name):
        self.last_name = last_name

    def set_company(self,company):
        self.company = company

    def set_title(self,title):
        self.title = title

    def set_dept_head3(self,dept_head3):
        self.dept_head3 = dept_head3

    def set_dept_head2(self,dept_head2):
        self.dept_head2 = dept_head2

    def set_dept_head(self,dept_head):
        self.dept_head = dept_head

    def set_front_back(self,front_back):
        self.front_back = front_back

    def set_supervisor(self,supervisor):
        self.supervisor = supervisor

    def set_front_office_sector(self,front_office_sector):
        self.front_office_sector = front_office_sector

    def set_date_of_hire(self,date_of_hire):
        self.date_of_hire = date_of_hire

    def set_strategy(self,strategy):
        self.strategy = strategy

    def set_employee_id(self,employee_id):
        self.employee_id = employee_id

    def set_tmt(self,tmt):
        self.tmt = tmt

    def set_gender(self,gender):
        self.gender = gender

    def set_location(self,location):
        self.location = location

    def set_subsidiary(self,subsidiary):
        self.subsidiary = subsidiary

    #############################################################
    ### Getter Functions
    #############################################################

    def get_name(self) -> str:
        """Gets the user name of the Employee.


        Returns:
            User name : last_name + first_name
         """

        self.user_name = self.last_name + "_" + self.first_name
        return self.user_name

    #############################################################
    ### Representation
    #############################################################
    def __repr__(self):
        return "\n [[User Name : {0}, Title : {1} Date of Hire : {2}\n  \
               Supervisor: {3}, Dept.of Head : {4} , {5} , {6} \n \
               front office sector : {7}]]".format(self.get_name(),self.title,self.date_of_hire,
                                                 self.supervisor,self.dept_head,self.dept_head2,self.dept_head3,
                                                 self.front_office_sector)
    def __str__(self):
        return "\n [[User Name : {0}, Title : {1} Date of Hire : {2}\n  \
               Supervisor: {3}, Dept.of Head : {4} , {5} , {6} \n \
               front office sector : {7}]]".format(self.get_name(),self.title,self.date_of_hire,
                                                 self.supervisor,self.dept_head,self.dept_head2,self.dept_head3,
                                                 self.front_office_sector)

def employee_id_to_username_from_file(file_path:str) -> Tuple[dict,dict]:
    """Returns employee id to employee name dictionary.
    Args:
        file_path : path to Employee Master File

    Returns:
        Dictionary
        Key :  Employee id
        Value : last_name + "_" + first_name
     """
    employee_id_to_username_dict = {}
    df = pd.read_excel(file_path)
    emp_id_list = df["el_id"].tolist()
    emp_name_list = (df["last.name"] + "_" +df["first.name"]).tolist()

    employee_id_to_username_dict = {key:value for key,value in zip(emp_id_list,emp_name_list)}
    employee_username_to_id_dict = {key: value for key, value in zip(emp_name_list,emp_id_list)}
    return employee_id_to_username_dict,employee_username_to_id_dict

def  get_emplpoyees_from_file(file_path : str) -> dict:
    """Returns employees dictionary by parsing the employee file.

    Args:
        file_path : path to file Employee Master(xlxs)

    Returns:
        Dictionary of Employees
        key : last_name + "_" + first_name
        value: Employee object
    """
    employee_dict = {}
    df = pd.read_excel(file_path)
    ## Fill NA with empty string.
    df = df.fillna("")
    employee_id_dict,_ = employee_id_to_username_from_file(file_path)
    gender_dict = {0:"M",1:"F"}

    for i in range(len(df)):
        emp = Employee()
        last_name = df.loc[i,"last.name"].strip()
        first_name = df.loc[i,"first.name"].strip()
        company = df.loc[i,"Company"].strip()
        title = df.loc[i,"Title"].strip()
        dept_head3 = employee_id_dict[df.loc[i,"Dept.Head3"].item()] ## getting the employee from id
        dept_head2 = df.loc[i,"Dept.Head2"]
        dept_head = df.loc[i,"Dept.Head"]
        front_back = df.loc[i,"Front_Back"]
        supervisor = df.loc[i, "Supervisor"]
        front_office_sector = df.loc[i,"Front.Office.Sector"]
        date_of_hire = df.loc[i,"Date.of.Hire"]
        strategy = df.loc[i,"Strategy"]
        employee_id = df.loc[i,"el_id"]
        tmt = df.loc[i,"TMT"]
        gender = gender_dict[df.loc[i,"female"]]
        location = df.loc[i,"Location"]
        subsidiary = df.loc[i,"Subsidiary"]

        ## Setting the employee variables
        emp.set_last_name(last_name); emp.set_first_name(first_name) ; emp.set_company(company)
        emp.set_title(title) ; emp.set_dept_head3(dept_head3) ; emp.set_dept_head2(dept_head2)
        emp.set_dept_head(dept_head) ; emp.set_front_back(front_back) ; emp.set_supervisor(supervisor)
        emp.set_front_office_sector(front_office_sector) ; emp.set_date_of_hire(date_of_hire)
        emp.set_strategy(strategy) ; emp.set_employee_id(employee_id) ; emp.set_tmt(tmt)
        emp.set_gender(gender) ; emp.set_location(location) ; emp.set_subsidiary(subsidiary)

        name = emp.get_name()
        employee_dict[name] = emp

    return  employee_dict

def map_user_address(file_path : str) -> Tuple[dict,dict]:
    """Maps different IM names to user names.

    Args:
        file_path : path to the address linker file.

    Return:
        Tuple of dictionaries.
        address_to_user_dict : key - address ; value - user name
        user_to_address_dict : key - user name ; value - address

    """

    f = open(file_path, "r")
    ## Each user will have multiple addresses. Therefore values will be list
    user_to_address_dict = {}
    ## Each address corresponds to a user. Therefore values will be userLastName_userFirstName
    address_to_user_dict = {}
    ## To skip reading the firstline
    f.readline()
    for line in f.readlines():
        line = line.lower()
        line_split = line.split()
        last_name = line_split[0]
        address = line_split[-1].split("@")[0]
        first_name = line_split[-2]
        name = last_name + "_"+ first_name
        address_to_user_dict[address] = name

        address_list = user_to_address_dict.get(name,[])
        address_list.append(address)
        user_to_address_dict[name] = address_list

    return address_to_user_dict,user_to_address_dict

def map_employee_account(file_path : str) -> Tuple[dict,dict]:
    """Maps Employee to account and account to employee.

    Args:
        file_path : Trader_Book_Account file Path

    Returns:
        Tuple of Dictionaries
        account_to_employee_dict : key - account ; value - employee name
        employee_to_account_dict : key - employee name ; value - account

    """
    df = pd.read_excel(file_path)
    employees_list = df["name"].tolist()
    account_list = df["Account"].tolist()
    account_to_employee_dict = {}
    employee_to_account_dict = {}



    for employee, account_string in zip(employees_list,account_list):
        ## name = Wolfberg.Adam need to convert to wolfberg_adam. To maintain consistency
        employee = employee.lower().replace(".","_")
        employee_to_account_dict[employee] = []
        for account in account_string.split(","):
            account = account.strip()
            account_to_employee_dict[account] = account_to_employee_dict.get(account,[])
            account_to_employee_dict[account].append(employee)
            employee_to_account_dict[employee].append(account)
            # print(account)

    return account_to_employee_dict,employee_to_account_dict



def lambda_func_user_address_mapping(address : str) -> str:
    """Lambda function that maps user address to user name.

    Args:
        Address : Address of the user

    Returns:
        User name : User name of the user(lastname_firstname)

    if address is present in the address user dictionary then returns user name else address
    """
    address = address.split("@")[0]
    try:
        user = address_to_user_dict[address]
        return user
    except:
        return address

def lambda_func_user_in_network(user_name : str) -> int:
    """Lambda function that returns 1/0 based on user being inside/outside network.

    Args:
        User name : User name of the user(lastname_firstname)

    Returns:
        int

    if user is present in the employee dict then return 1 else 0.
    """

    try:
        emp = employee_dict[user_name] ## faster to check dictionary than iterating over list.
        return 1
    except:
        return 0

# =========================================================================
# ==================== hierarchy structure=================================
# =========================================================================

def create_employee_hierarchy(employee_dict : dict) -> dict:
    """Created hierarchy of Employees.

    Args:
        employee_dict : Employee Dictionary.

    Returns:
          Employee dictionary with top employee included.

    """
    ## Create Top Employee in the hierarchy, such that its subordinates are heads of individual departments.
    top_employee = Employee()
    top_employee.user_name = "ROOT"
    top_employee.title = "ROOT"
    for employee_name, employee in employee_dict.items():
        supervisor = employee.supervisor

        if(supervisor != ""): ## since NA is filled with empty string.
            ## To convert "Wade, Peter -> wade_peter"
            supervisor = supervisor.split('/')[0] ## for now taking just one.
            supervisor = supervisor.split(',') ; supervisor = [x.strip().lower() for x in supervisor]
            supervisor = "_".join(supervisor)
            employee_dict[supervisor].immediate_subordinates.append(employee_name)

        else:
            # print("supervisor not availbale : {0}".format(employee_name))
            top_employee.immediate_subordinates.append(employee_name)

    employee_dict["ROOT"] = top_employee
    return employee_dict

def subordinates_given_employee(employee_dict:dict, inp_employee_name:str) -> List:
    """Returns all the subordinates of given employee including employee himself.

    Args:
        employee_dict: Employee dictionary after creating hierarchy.
        inp_employee_name : Employee name whose subordinates are returned.

    Returns:
        subordinates_list : subordinates list of given employee.
    """
    subordinates_list = []
    employee_queue = queue.Queue()
    employee_queue.put(inp_employee_name)

    while (not employee_queue.empty()):
        curr_emp_name = employee_queue.get()
        subordinates_list.append(curr_emp_name)
        for subordinate in employee_dict[curr_emp_name].immediate_subordinates:
            employee_queue.put(subordinate)
    subordinates_list = sorted(subordinates_list)
    return subordinates_list

# =========================================================================
# ==================== Add top user in hierarchy for each user.============
# =========================================================================

def compute_top_user_for_each_user(employee_dict : dict) -> dict:
    """Creates top user for each user present in the hedgefund.

    Args:
        employee_dict : Employee Dictionary.

    Returns:
          Employee dictionary with top employee for each user included.

    """
    for employee_name, employee in employee_dict.items():
        # print(employee_name)
        if(employee_name != "ROOT"):
            curr_supervisor = employee.supervisor ; prev_supervisor = employee_name
            while curr_supervisor != "ROOT":
                if (curr_supervisor != ""):
                    ## To convert "Wade, Peter -> wade_peter"
                    curr_supervisor = curr_supervisor.split('/')[0]  ## for now taking just one.
                    curr_supervisor = curr_supervisor.split(',');
                    curr_supervisor = [x.strip().lower() for x in curr_supervisor]
                    curr_supervisor = "_".join(curr_supervisor)

                    prev_supervisor = curr_supervisor
                    curr_supervisor = employee_dict[curr_supervisor].supervisor
                else:
                    break
        else:
            break

        employee.top_user_in_hierarchy = prev_supervisor

    return employee_dict


# =========================================================================
# ==================== Books given employee List===========================
# =========================================================================

def books_given_employee_list(inp_employee_list : List) -> List:
    """Given a list of users, returns the list of books associated with them.

    Args:
        inp_employee_list : Input Employee List.

    Returns:
        book_list : books associated with employees.

    """
    book_set = set()
    for employee_name in inp_employee_list:
        if employee_name in employee_to_account_dict.keys():
            for book in employee_to_account_dict[employee_name]:
                book_set.add(book)

    book_list = sorted(list(book_set))
    return book_list

# =========================================================================
# ==================== Employees given Book List===========================
# =========================================================================

def employees_given_book_list(inp_book_list : List) -> List:
    """Given a list of books, returns the list of employees associated with them.

    Args:
        inp_book_list : Input Book List.

    Returns:
        emp_list : Employees associated with books

    """
    employee_set = set()
    for book_name in inp_book_list:
        if book_name in account_to_employee_dict.keys():
            for employee in account_to_employee_dict[book_name]:
                employee_set.add(employee)

    emp_list = sorted(list(employee_set))
    return emp_list


# =========================================================================
# ==================== Compute necessary dictionaries =====================
# =========================================================================
address_to_user_dict,user_to_address_dict = map_user_address(cfg.ADDRESS_LINK_FILE)
employee_dict = get_emplpoyees_from_file(cfg.EMPLOYEE_MASTER_FILE)
employee_list = list(employee_dict.keys())
account_to_employee_dict,employee_to_account_dict = map_employee_account(cfg.TRADER_BOOK_ACCOUNT_FILE)
employee_dict = create_employee_hierarchy(employee_dict)
employee_dict = compute_top_user_for_each_user(employee_dict)
employee_id_to_username_dict,employee_username_to_id_dict = employee_id_to_username_from_file(cfg.EMPLOYEE_MASTER_FILE)
# =========================================================================
