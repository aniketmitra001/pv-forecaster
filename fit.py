import models
import loaders
import argparse
import torch




if __name__ == "__main__":

    parser = argparse.ArgumentParser("Model for day-ahead PV forecasting")
    parser.add_argument('--input_file', type=str,
                        help="Path to the input fie")
    parser.add_argument('--scaler', default=0.01,type=float,
                        help="Scaling factor to normalize the power output readings")  
    parser.add_argument('--min_model_input_size', default=96,type=int,
                        help="Min size of the model input")
    parser.add_argument('--max_model_input_size', default=96,type=int,
                        help="Max size of the model input")  
    parser.add_argument('--min_model_output_size', default=96,type=int,
                        help="Min size of the model output")
    parser.add_argument('--max_model_output_size', default=96,type=int,
                        help="Max size of the model output") 
    parser.add_argument('--hidden_layer_size', default=10,type=int,
                        help="Size of each hidden layer in the Feed Forward Neural Network")
    parser.add_argument('--num_hidden_layers', default=2,type=int,
                        help="Number of hidden layers in the Feed Forward Neural Network")  
    parser.add_argument("--num_steps", default = 500, type=int,
                        help="Number of steps to optimize the weights")
    parser.add_argument("--learning_rate", default = 0.001, type=int,
                        help="the step size of learning algorithm")
    parser.add_argument("--filter_criteria",  type=int,
                        help="Filter data for a given year")      
    parser.add_argument("--output_file",type=str,
                        help="Name of the file where the model will be saved")                                                                                              
    args = parser.parse_args() 
    
    dataset = loaders.PVData(args.input_file)
    dataset.scale_power_output(args.scaler)
    
   
    encoder_length = {'min' : args.min_model_input_size,
                    'max' :  args.max_model_input_size }

    prediction_length = {'min' : args.min_model_output_size, 
                    'max':  args.max_model_output_size}
  
  
    training_set = dataset.get_examples(args.filter_criteria, encoder_length, prediction_length)
    print('Training Set:',training_set.get_parameters())

    models.FullyConnectedModelWithCovariates.from_dataset(training_set, 
            hidden_size=args.hidden_layer_size, n_hidden_layers=args.num_hidden_layers, learning_rate = args.learning_rate)
   

    model = models.FullyConnectedModelWithCovariates.optimize(training_set, num_steps=args.num_steps,
             hidden_size=args.hidden_layer_size, n_hidden_layers=args.num_hidden_layers, learning_rate = args.learning_rate)

    model.summarize("full")  # print model summary
 
    print(model.eval())

    print(model.hparams)         


    torch.save(model, args.output_file)