#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:31:52 2021

@author: ljia
"""
import os
import sys
import urllib
# import requests
import pandas as pd
from lxml import etree
from tqdm import tqdm


def fetch_dataset(ds_name, **kwargs):
	if ds_name.lower() == 'thermophysical':
		fname = fetch_thermophysical(**kwargs)

	return fname


#%%


def fetch_thermophysical(fname='../datasets/Thermophysical/thermophysical.csv'):

	url = 'https://polymerdatabase.com/home.html'

	# Get data from the website.
	df_main = _process_a_class_page(url)

	# Save dataset to file.
	os.makedirs(os.path.dirname(fname), exist_ok=True)
	df_main.to_csv(fname)

	return fname


def _process_a_class_page(url):

	### Create df to store data.
	df_page = pd.DataFrame(columns=['name', 'smiles', 'curlysmiles', 'Tg-range (exp)', 'Tg-preferred (exp)', 'Tg-range (cal)', 'Tg-preferred (cal)', 'class'])


	### Get data from the page.

	try:
		response = urllib.request.urlopen(url.replace(' ', '%20'))
	except urllib.error.HTTPError:
		print('The URL is not available:\n' + url)
		return df_page

	# Get homepage sources.
	page_str = response.read()
	p_tree = etree.HTML(page_str)

	# Get data from the page.
	df_cur_p = _get_data_from_a_class_page(p_tree, url)
	df_page = df_page.append(df_cur_p)


	### Go to the next page.
	a_node = p_tree.xpath('//li/a')[-1]
	if a_node.text is not None and ('Next' in a_node.text.strip() or 'More' in a_node.text.strip()):
		next_url = a_node.attrib['href'].strip()
		next_url = '/'.join(url.split('/')[:-1]) + '/' + next_url
		df_next_p = _process_a_class_page(next_url)
		df_page = df_page.append(df_next_p)
	# Reach the last page. # @todo: There might be errors instead of the last page.
	else:
		pass
# 		from urllib.error import URLError
# 		raise URLError('"This is neither a "Next" nor "More" button! Why is that?')


	return df_page


def _get_data_from_a_class_page(tree, url):

	print('\nClass list page: ', url)

	### Create df to store data.
	df_page = pd.DataFrame(columns=['name', 'smiles', 'curlysmiles', 'Tg-range (exp)', 'Tg-preferred (exp)', 'Tg-range (cal)', 'Tg-preferred (cal)', 'class'])


	# Get table.
	table = tree.xpath('//table')[0]

	# Get each line in the table.
	tr_nodes = table.xpath('tr')
	for tr in tqdm(tr_nodes[1:], desc='Retrieving data from classes', file=sys.stdout): # @todo: to change back
		# Get each element in the line.
		td_node = tr.xpath('td')[0]

		# class link.
		cls_url = td_node.xpath('.//a')[0].attrib['href'].strip()
		cls_url = '/'.join(url.split('/')[:-1]) + '/' + cls_url
		df_class = _get_data_in_the_class(cls_url)
		df_page = df_page.append(df_class)

	return df_page


def _get_data_in_the_class(url):

	print('Inside the class page: ', url)

	### Create df to store data.
	df_page = pd.DataFrame(columns=['name', 'smiles', 'curlysmiles', 'Tg-range (exp)', 'Tg-preferred (exp)', 'Tg-range (cal)', 'Tg-preferred (cal)', 'class'])


	### Get data from the page.

	try:
		response = urllib.request.urlopen(url.replace(' ', '%20')) # Tackle whitespaces in the url.
	except urllib.error.HTTPError:
		print('The URL is not available:\n' + url)
		return df_page

	# Get homepage sources.
	page_str = response.read()
	p_tree = etree.HTML(page_str)

# 	# Get data from the page.
# 	df_cur_p = _get_data_from_a_mol_page(p_tree)
# 	df_page = df_page.append(df_cur_p)


	# Get table.
	ul_node = p_tree.xpath('//ul')[-1]

	# Get each line in the ul.
	li_nodes = ul_node.xpath('li')
	for li in li_nodes[:]:
		# Get link to the mol page.
		mol_url = li.xpath('a')[0].attrib['href'].strip()
		mol_url = '/'.join(url.split('/')[:-1]) + '/' + mol_url
		mol_info = _get_mol_info_from_page(mol_url)
		if mol_info:
			df_page.loc[len(df_page)] = mol_info


	### Check if there are next pages.
	a_node = p_tree.xpath('//li/a')[-1]
	if a_node.text.strip() == 'Next' or a_node.text.strip() == 'More':
		raise Exception('I have next page! Who can imagine that! Check me out:\n' + url)


	return df_page


def _get_mol_info_from_page(url):

	print('Mol page: ', url)


	### Get data from the page.

	try:
		response = urllib.request.urlopen(url.replace(' ', '%20'))
	except urllib.error.HTTPError:
		print('The URL is not available:\n' + url)
		return False

	# Get homepage sources.
	page_str = response.read()
	p_tree = etree.HTML(page_str)

	# Get table.
	tables = p_tree.xpath('//table')

	# The first table.
	tr_nodes = tables[0].xpath('tbody/tr')
	class_ = tr_nodes[0].xpath('td')[1].text.strip()
	name = tr_nodes[1].xpath('td')[1].text.strip()
	curly_smiles = tr_nodes[5].xpath('td')[1].text
	if curly_smiles is None:
		curly_smiles = ''
	else:
		curly_smiles = curly_smiles.strip()

	# The second table.
	tr_nodes = tables[1].xpath('tbody/tr')
	smiles = tr_nodes[2].xpath('td')[1].text.strip()

	# The third table.
# 	caption = tables[2].getprevious().xpath('.//text()')[1]
# 	if 'Experimental' in caption:
	tr_nodes = tables[2].xpath('tbody/tr')
	Tg_range_exp = tr_nodes[5].xpath('td')[2].text
	Tg_preferred_exp = tr_nodes[5].xpath('td')[3].text
	# if Tg_range is a number rather than a range:
	if Tg_range_exp is not None and '-' not in Tg_range_exp and ',' not in Tg_range_exp:
		if Tg_preferred_exp is None:
			Tg_preferred_exp = Tg_range_exp
		Tg_range_exp = ''
	# else:
	else:
		Tg_range_exp = ('' if Tg_range_exp is None else _mystrip((Tg_range_exp)))
	Tg_preferred_exp = ('' if Tg_preferred_exp is None else _mystrip(Tg_preferred_exp))
# 	if Tg_preferred_exp != '':
# 		Tg_preferred_exp = float(Tg_preferred_exp)
# 		except Exception as e:
# 			print('Tg_preferred_exp: ', Tg_preferred_exp)
# 			print(Tg_preferred_exp == '')
# 			raise

	# The fourth table.
	if len(tables) > 3:
		tr_nodes = tables[3].xpath('tbody/tr')
		Tg_range_cal = tr_nodes[7].xpath('td')[2].text
		Tg_preferred_cal = tr_nodes[7].xpath('td')[3].text
		# if Tg_range is a number rather than a range:
		if Tg_range_cal is not None and '-' not in Tg_range_cal and ',' not in Tg_range_cal:
			if Tg_preferred_cal is None:
				Tg_preferred_cal = Tg_range_cal
			Tg_range_cal = ''
		# else:
		else:
			Tg_range_cal = ('' if Tg_range_cal is None else _mystrip(Tg_range_cal))
		Tg_preferred_cal = ('' if Tg_preferred_cal is None else _mystrip(Tg_preferred_cal))
	# 	if Tg_preferred_cal != '':
	# 		Tg_preferred_cal = float(Tg_preferred_cal)
	else:
		Tg_range_cal, Tg_preferred_cal = '', ''



	return [name, smiles, curly_smiles, Tg_range_exp, Tg_preferred_exp, Tg_range_cal, Tg_preferred_cal, class_]


def _mystrip(str_):
	return str_.strip().lstrip('(').rstrip(')').strip()