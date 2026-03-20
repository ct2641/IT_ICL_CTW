# 
# Reference arithmetic coding
# 
# Copyright (c) Project Nayuki
# MIT License. See readme file.
# https://www.nayuki.io/page/reference-arithmetic-coding
# 

# ---- Frequency table classes ----

# A table of symbol frequencies. The table holds data for symbols numbered from 0
# to get_symbol_limit()-1. Each symbol has a frequency, which is a non-negative integer.
# Frequency table objects are primarily used for getting cumulative symbol
# frequencies. These objects can be mutable depending on the implementation.
class FrequencyTable:
	
	# Returns the number of symbols in this frequency table, which is a positive number.
	def get_symbol_limit(self):
		raise NotImplementedError()
	
	# Returns the frequency of the given symbol. The returned value is at least 0.
	def get(self, symbol):
		raise NotImplementedError()
	
	# Sets the frequency of the given symbol to the given value.
	# The frequency value must be at least 0.
	def set(self, symbol, freq):
		raise NotImplementedError()
	
	# Increments the frequency of the given symbol.
	def increment(self, symbol):
		raise NotImplementedError()
	
	# Returns the total of all symbol frequencies. The returned value is at
	# least 0 and is always equal to get_high(get_symbol_limit() - 1).
	def get_total(self):
		raise NotImplementedError()
	
	# Returns the sum of the frequencies of all the symbols strictly
	# below the given symbol value. The returned value is at least 0.
	def get_low(self, symbol):
		raise NotImplementedError()
	
	# Returns the sum of the frequencies of the given symbol
	# and all the symbols below. The returned value is at least 0.
	def get_high(self, symbol):
		raise NotImplementedError()



# An immutable frequency table where every symbol has the same frequency of 1.
# Useful as a fallback model when no statistics are available.
class FlatFrequencyTable(FrequencyTable):
	
	# Constructs a flat frequency table with the given number of symbols.
	def __init__(self, numsyms):
		if numsyms < 1:
			raise ValueError("Number of symbols must be positive")
		self.numsymbols = numsyms  # Total number of symbols, which is at least 1
	
	# Returns the number of symbols in this table, which is at least 1.
	def get_symbol_limit(self):
		return self.numsymbols
	
	# Returns the frequency of the given symbol, which is always 1.
	def get(self, symbol):
		self._check_symbol(symbol)
		return 1
	
	# Returns the total of all symbol frequencies, which is
	# always equal to the number of symbols in this table.
	def get_total(self):
		return self.numsymbols
	
	# Returns the sum of the frequencies of all the symbols strictly below
	# the given symbol value. The returned value is equal to 'symbol'.
	def get_low(self, symbol):
		self._check_symbol(symbol)
		return symbol
	
	
	# Returns the sum of the frequencies of the given symbol and all
	# the symbols below. The returned value is equal to 'symbol' + 1.
	def get_high(self, symbol):
		self._check_symbol(symbol)
		return symbol + 1
	
	
	# Returns silently if 0 <= symbol < numsymbols, otherwise raises an exception.
	def _check_symbol(self, symbol):
		if not (0 <= symbol < self.numsymbols):
			raise ValueError("Symbol out of range")
	
	# Returns a string representation of this frequency table. The format is subject to change.
	def __str__(self):
		return "FlatFrequencyTable={}".format(self.numsymbols)
	
	# Unsupported operation, because this frequency table is immutable.
	def set(self, symbol, freq):
		raise NotImplementedError()
	
	# Unsupported operation, because this frequency table is immutable.
	def increment(self, symbol):
		raise NotImplementedError()



# A mutable table of symbol frequencies. The number of symbols cannot be changed
# after construction. The current algorithm for calculating cumulative frequencies
# takes linear time, but there exist faster algorithms such as Fenwick trees.
class SimpleFrequencyTable(FrequencyTable):
	
	# Constructs a simple frequency table in one of two ways:
	# - SimpleFrequencyTable(sequence):
	#   Builds a frequency table from the given sequence of symbol frequencies.
	#   There must be at least 1 symbol, and no symbol has a negative frequency.
	# - SimpleFrequencyTable(freqtable):
	#   Builds a frequency table by copying the given frequency table.
	def __init__(self, freqs):
		if isinstance(freqs, FrequencyTable):
			numsym = freqs.get_symbol_limit()
			self.frequencies = [freqs.get(i) for i in range(numsym)]
		else:  # Assume it is a sequence type
			self.frequencies = list(freqs)  # Make copy
		
		# 'frequencies' is a list of the frequency for each symbol.
		# Its length is at least 1, and each element is non-negative.
		if len(self.frequencies) < 1:
			raise ValueError("At least 1 symbol needed")
		for freq in self.frequencies:
			if freq < 0:
				raise ValueError("Negative frequency")
		
		# Always equal to the sum of 'frequencies'
		self.total = sum(self.frequencies)
		
		# cumulative[i] is the sum of 'frequencies' from 0 (inclusive) to i (exclusive).
		# Initialized lazily. When it is not None, the data is valid.
		self.cumulative = None
	
	
	# Returns the number of symbols in this frequency table, which is at least 1.
	def get_symbol_limit(self):
		return len(self.frequencies)
	
	
	# Returns the frequency of the given symbol. The returned value is at least 0.
	def get(self, symbol):
		self._check_symbol(symbol)
		return self.frequencies[symbol]
	
	
	# Sets the frequency of the given symbol to the given value. The frequency value
	# must be at least 0. If an exception is raised, then the state is left unchanged.
	def set(self, symbol, freq):
		self._check_symbol(symbol)
		if freq < 0:
			raise ValueError("Negative frequency")
		temp = self.total - self.frequencies[symbol]
		assert temp >= 0
		self.total = temp + freq
		self.frequencies[symbol] = freq
		self.cumulative = None
	
	
	# Increments the frequency of the given symbol.
	def increment(self, symbol):
		self._check_symbol(symbol)
		self.total += 1
		self.frequencies[symbol] += 1
		self.cumulative = None
	
	
	# Returns the total of all symbol frequencies. The returned value is at
	# least 0 and is always equal to get_high(get_symbol_limit() - 1).
	def get_total(self):
		return self.total
	
	
	# Returns the sum of the frequencies of all the symbols strictly
	# below the given symbol value. The returned value is at least 0.
	def get_low(self, symbol):
		self._check_symbol(symbol)
		if self.cumulative is None:
			self._init_cumulative()
		return self.cumulative[symbol]
	
	
	# Returns the sum of the frequencies of the given symbol
	# and all the symbols below. The returned value is at least 0.
	def get_high(self, symbol):
		self._check_symbol(symbol)
		if self.cumulative is None:
			self._init_cumulative()
		return self.cumulative[symbol + 1]
	
	
	# Recomputes the array of cumulative symbol frequencies.
	def _init_cumulative(self):
		cumul = [0]
		sum = 0
		for freq in self.frequencies:
			sum += freq
			cumul.append(sum)
		assert sum == self.total
		self.cumulative = cumul
	
	
	# Returns silently if 0 <= symbol < len(frequencies), otherwise raises an exception.
	def _check_symbol(self, symbol):
		if not (0 <= symbol < len(self.frequencies)):
			raise ValueError("Symbol out of range")
	
	
	# Returns a string representation of this frequency table,
	# useful for debugging only, and the format is subject to change.
	def __str__(self):
		result = ""
		for (i, freq) in enumerate(self.frequencies):
			result += "{}\t{}\n".format(i, freq)
		return result



# A wrapper that checks the preconditions (arguments) and postconditions (return value) of all
# the frequency table methods. Useful for finding faults in a frequency table implementation.
class CheckedFrequencyTable(FrequencyTable):
	
	def __init__(self, freqtab):
		# The underlying frequency table that holds the data
		self.freqtable = freqtab
	
	
	def get_symbol_limit(self):
		result = self.freqtable.get_symbol_limit()
		if result <= 0:
			raise AssertionError("Non-positive symbol limit")
		return result
	
	
	def get(self, symbol):
		result = self.freqtable.get(symbol)
		if not self._is_symbol_in_range(symbol):
			raise AssertionError("ValueError expected")
		if result < 0:
			raise AssertionError("Negative symbol frequency")
		return result
	
	
	def get_total(self):
		result = self.freqtable.get_total()
		if result < 0:
			raise AssertionError("Negative total frequency")
		return result
	
	
	def get_low(self, symbol):
		if self._is_symbol_in_range(symbol):
			low   = self.freqtable.get_low (symbol)
			high  = self.freqtable.get_high(symbol)
			if not (0 <= low <= high <= self.freqtable.get_total()):
				raise AssertionError("Symbol low cumulative frequency out of range")
			return low
		else:
			self.freqtable.get_low(symbol)
			raise AssertionError("ValueError expected")
	
	
	def get_high(self, symbol):
		if self._is_symbol_in_range(symbol):
			low   = self.freqtable.get_low (symbol)
			high  = self.freqtable.get_high(symbol)
			if not (0 <= low <= high <= self.freqtable.get_total()):
				raise AssertionError("Symbol high cumulative frequency out of range")
			return high
		else:
			self.freqtable.get_high(symbol)
			raise AssertionError("ValueError expected")
	
	
	def __str__(self):
		return "CheckedFrequencyTable (" + str(self.freqtable) + ")"
	
	
	def set(self, symbol, freq):
		self.freqtable.set(symbol, freq)
		if not self._is_symbol_in_range(symbol) or freq < 0:
			raise AssertionError("ValueError expected")
	
	
	def increment(self, symbol):
		self.freqtable.increment(symbol)
		if not self._is_symbol_in_range(symbol):
			raise AssertionError("ValueError expected")
	
	
	def _is_symbol_in_range(self, symbol):
		return 0 <= symbol < self.get_symbol_limit()

