from main_booknlp import BookNLP

model_params={
		"pipeline":"entity,quote,supersense,event,coref", 
		"model":"big"
	}
	
booknlp=BookNLP("en", model_params)

# Input file to process
input_file="examples/158_emma.txt"

# Output directory to store resulting files in
output_directory="emma_output/"

# File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.
book_id="emma"

booknlp.process(input_file, output_directory, book_id)