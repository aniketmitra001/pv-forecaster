import models
import loaders
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model for day-ahead PV forecasting")
    parser.add_argument('--input_file', type=str,
                        help="Path to the input file")
    parser.add_argument('--model', type=str,
                        help="Path to the model")                    
    parser.add_argument('--scaler', default=0.01,type=float,
                        help="Scaling factor to normalize the power output readings")  
    parser.add_argument("--filter_criteria",  type=int,
                        help="Filter data for a given year")  

    args = parser.parse_args()                     

    model = torch.load(args.model)

    dataset = loaders.PVData(args.input_file)
    dataset.scale_power_output(args.scaler)

    print(model.hparams)

    encoder_length = {'min' : model.hparams['input_size'] ,
                    'max' :  model.hparams['input_size']}

    prediction_length = {'min' : model.hparams['output_size'], 
                    'max':  model.hparams['output_size']}
  
  
    validation_set = dataset.get_examples(args.filter_criteria, encoder_length, prediction_length)
    print('Validation Set:',validation_set.get_parameters())

    dataloader = validation_set.to_dataloader()
    criteria = torch.nn.L1Loss(reduction='sum') 
    total_loss = 0
    num_elems = 0
    for x_test,y_test in dataloader:
        y_pred = model(x_test)['prediction']
        loss = criteria(y_pred , y_test[0])
        total_loss += loss
        num_elems += (y_pred.shape[0] * y_pred.shape[1])
        for pred_time_tensor,prediction_tensor in zip(x_test['decoder_time_idx'],y_pred):
            for x,y in zip(pred_time_tensor,prediction_tensor):
                print('Time_idx {}:, Prediction: {}'.format(x,y / args.scaler))
    
    print('Sum Absolute Error {}:, Num Prediction: {}, MAE: {}'.format(total_loss,num_elems,total_loss /  num_elems))
  


