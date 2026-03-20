# 
# Reference arithmetic coding
# 
# Copyright (c) Project Nayuki
# MIT License. See readme file.
# https://www.nayuki.io/page/reference-arithmetic-coding
# 

from frequency_table import FlatFrequencyTable, SimpleFrequencyTable
import torch

class PpmModel:
	
	def __init__(self, order, symbollimit, escapesymbol):
		if not ((order >= -1) and (0 <= escapesymbol < symbollimit)):
			raise ValueError()
		self.model_order = order
		self.symbol_limit = symbollimit
		self.escape_symbol = escapesymbol
		
		if order >= 0:
			self.root_context = PpmModel.Context(symbollimit, order >= 1,self.model_order)
			self.root_context.frequencies.increment(escapesymbol)
		else:
			self.root_context = None
		self.order_minus1_freqs = FlatFrequencyTable(symbollimit)
	
	
	def increment_contexts(self, history, symbol):
		if self.model_order == -1:
			return
		if not ((len(history) <= self.model_order) and (0 <= symbol < self.symbol_limit)):
			raise ValueError()
		
		ctx = self.root_context
		ctx.frequencies.increment(symbol)
		for (i, sym) in enumerate(history):
			subctxs = ctx.subcontexts
			assert subctxs is not None
			
			if subctxs[sym] is None:
				subctxs[sym] = PpmModel.Context(self.symbol_limit, i + 1 < self.model_order,self.model_order)				
				# # test to see if we use method A
				# if subctxs[sym].frequencies.get(self.escape_symbol) == 0:
				# 	subctxs[sym].frequencies.increment(self.escape_symbol)
				# elif subctxs[sym].frequencies.get(self.escape_symbol) == 0:
				# 	subctxs[sym].frequencies.set(self.escape_symbol,0)
				subctxs[sym].frequencies.increment(self.escape_symbol)
			ctx = subctxs[sym]
			ctx.frequencies.increment(symbol)
	
	# Helper structure
	class Context:
		
		def __init__(self, symbols, hassubctx, depth):
			self.frequencies = SimpleFrequencyTable([0] * symbols)
			if not hassubctx:
				self.err = torch.zeros(depth+1) 
			self.subcontexts = ([None] * symbols) if hassubctx else None

		def err_increment(self,value):
			self.err += value