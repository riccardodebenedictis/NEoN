/*
 * Copyright (C) 2018 Riccardo De Benedictis
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package it.cnr.istc.pst.neon;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.swing.JFrame;
import javax.swing.JProgressBar;

import com.google.gson.Gson;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.time.Millisecond;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;

/**
 *
 * @author Riccardo De Benedictis
 */
public class App {

    private static final String START_TRAINING = "strt_tr";
    private static final String STOP_TRAINING = "stp_tr";
    private static final String START_EPOCH = "strt_epc";
    private static final String STOP_EPOCH = "stp_epc";

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        TimeSeries tr_data_error = new TimeSeries("Training data");
        TimeSeries tst_data_error = new TimeSeries("Test data");
        TimeSeriesCollection collection = new TimeSeriesCollection();
        collection.addSeries(tr_data_error);
        collection.addSeries(tst_data_error);

        JFrame frame = new JFrame("NEoN");
        frame.setPreferredSize(new Dimension(800, 600));
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        JFreeChart chart = ChartFactory.createTimeSeriesChart("Error", "", "Error", collection);
        frame.add(new ChartPanel(chart), BorderLayout.CENTER);

        JProgressBar bar = new JProgressBar();
        bar.setStringPainted(true);
        frame.add(bar, BorderLayout.SOUTH);

        frame.setVisible(true);

        Gson gson = new Gson();
        System.out.println("waiting for data..");
        try (ServerSocket serverSocket = new ServerSocket(1100)) {
            Socket clientSocket = serverSocket.accept();
            BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream(), "UTF-8"));
            String inputLine;
            while ((inputLine = in.readLine()) != null) {
                if (inputLine.startsWith(START_TRAINING)) {
                    StartTraining strt_tr = gson.fromJson(inputLine.substring(START_TRAINING.length()),
                            StartTraining.class);
                    bar.setValue(0);
                    bar.setMaximum(strt_tr.n_epcs);
                    tr_data_error.addOrUpdate(new Millisecond(), strt_tr.tr_error);
                    tst_data_error.addOrUpdate(new Millisecond(), strt_tr.tst_error);
                } else if (inputLine.startsWith(STOP_TRAINING)) {
                    StopTraining stp_tr = gson.fromJson(inputLine.substring(STOP_TRAINING.length()),
                            StopTraining.class);
                    tr_data_error.addOrUpdate(new Millisecond(), stp_tr.tr_error);
                    tst_data_error.addOrUpdate(new Millisecond(), stp_tr.tst_error);
                } else if (inputLine.startsWith(START_EPOCH)) {
                    StartEpoch strt_e = gson.fromJson(inputLine.substring(START_EPOCH.length()), StartEpoch.class);
                    tr_data_error.addOrUpdate(new Millisecond(), strt_e.tr_error);
                    tst_data_error.addOrUpdate(new Millisecond(), strt_e.tst_error);
                } else if (inputLine.startsWith(STOP_EPOCH)) {
                    StopEpoch stp_e = gson.fromJson(inputLine.substring(STOP_EPOCH.length()), StopEpoch.class);
                    bar.setValue(bar.getValue() + 1);
                    tr_data_error.addOrUpdate(new Millisecond(), stp_e.tr_error);
                    tst_data_error.addOrUpdate(new Millisecond(), stp_e.tst_error);
                }
            }
        } catch (IOException ex) {
            Logger.getLogger(App.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private static class StartTraining {
        private int n_epcs; // the total number of epochs..
        private double tr_error; // the initial error on training data..
        private double tst_error; // the initial error on test data..
    }

    private static class StopTraining {
        private double tr_error; // the final error on training data..
        private double tst_error; // the final error on test data..
    }

    private static class StartEpoch {
        private double tr_error; // the initial error on training data..
        private double tst_error; // the initial error on test data..
    }

    private static class StopEpoch {
        private double tr_error; // the final error on training data..
        private double tst_error; // the final error on test data..
    }
}
