/*
 *  Cypress -- C++ Spiking Neural Network Simulation Framework
 *  Copyright (C) 2019 Christoph Ostrau
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "brainscales.hpp"

#include <algorithm>
//#include <boost/make_shared.hpp>
#include <csignal>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

// Cypress Related ____________________________
#include <cypress/backend/resources.hpp>
#include <cypress/core/backend.hpp>
#include <cypress/core/exceptions.hpp>
#include <cypress/core/network.hpp>
#include <cypress/core/network_base.hpp>
#include <cypress/core/network_base_objects.hpp>
#include <cypress/core/neurons.hpp>
#include <cypress/util/logger.hpp>

// BrainScaleS related ________________________
#include "euter/objectstore.h"

// Population related
#include "euter/celltypes.h"
#include "euter/population.h"
#include "euter/population_view.h"
//#include "euter/nativerandomgenerator.h"

// Connector
#include "euter/alltoallconnector.h"
#include "euter/connector.h"
#include "euter/connector_types.h"
#include "euter/fixednumberpostconnector.h"
#include "euter/fixednumberpreconnector.h"
#include "euter/fixedprobabilityconnector.h"
#include "euter/fromlistconnector.h"
#include "euter/nativerandomgenerator.h"
#include "euter/onetooneconnector.h"
#include "euter/projection.h"

// logger
#include <log4cxx/basicconfigurator.h>

// marocco
#include "hal/Coordinate/iter_all.h"
#include "pymarocco/PyMarocco.h"
#include "pymarocco/runtime/Runtime.h"
#include "sthal/HICANN.h"
#include "sthal/Wafer.h"

// submit experiments to the wafer
#include "submit.h"

using boost::make_shared;
namespace C = HMF::Coordinate;
namespace D = halco::hicann::v2;
/**
 * Setting low lever parameter for the BrainScaleS System
 *
 * @param wafer pointer to wafer object
 * @param gmax set to 1023
 * @param gmax_div set to 2
 */
void cypress::BrainScaleS::set_stahl_params(
    boost::shared_ptr<::sthal::Wafer> wafer, double gmax, double gmax_div)
{
	for (auto hicann : wafer->getAllocatedHicannCoordinates()) {
		auto fgs = (*wafer)[hicann].floating_gates;

		for (auto block : C::iter_all<C::FGBlockOnHICANN>()) {
			fgs.setShared(block, HMF::HICANN::shared_parameter::V_gmax0, gmax);
			fgs.setShared(block, HMF::HICANN::shared_parameter::V_gmax1, gmax);
			fgs.setShared(block, HMF::HICANN::shared_parameter::V_gmax2, gmax);
			fgs.setShared(block, HMF::HICANN::shared_parameter::V_gmax3, gmax);
		}

		for (auto driver : C::iter_all<D::SynapseDriverOnHICANN>()) {
			for (auto row : C::iter_all<C::RowOnSynapseDriver>()) {
				(*wafer)[hicann].synapses[driver][row].set_gmax_div(
				    HMF::HICANN::GmaxDiv(gmax_div));
			}
		}

		for (size_t i = 0; i < fgs.getNoProgrammingPasses(); i++) {
			auto cfg = fgs.getFGConfig(C::Enum(i));
			cfg.fg_biasn = 0;
			cfg.fg_bias = 0;
			fgs.setFGConfig(C::Enum(i), cfg);
		}
		for (auto block : C::iter_all<C::FGBlockOnHICANN>()) {
			fgs.setShared(block, HMF::HICANN::shared_parameter::V_dllres, 275);
			fgs.setShared(block, HMF::HICANN::shared_parameter::V_ccas, 800);
		}
	}
}
/**
 * Init some of the loggers
 */
void cypress::BrainScaleS::init_logger()
{
	log4cxx::BasicConfigurator::resetConfiguration();
	log4cxx::BasicConfigurator::configure();

	auto level = global_logger().min_level();
	auto new_level = log4cxx::Level::getInfo();
	if (level == LogSeverity::DEBUG) {
		new_level = log4cxx::Level::getDebug();
	}
	else if (level == LogSeverity::INFO) {
		new_level = log4cxx::Level::getInfo();
	}
	else if (level == LogSeverity::WARNING) {
		new_level = log4cxx::Level::getWarn();
	}
	else if (level == LogSeverity::ERROR) {
		new_level = log4cxx::Level::getError();
	}
	else if (level == LogSeverity::FATAL_ERROR) {
		new_level = log4cxx::Level::getFatal();
	}

	auto logger = log4cxx::Logger::getRootLogger();
	logger->setLevel(new_level);
	for (auto logger :
	     {"ESS", "marocco", "calibtic", "stahl", "Default", "halbe"}) {
		auto logger_inst = log4cxx::Logger::getLogger(logger);
		logger_inst->setLevel(new_level);
	}
}

cypress::BrainScaleS::BrainScaleS(const Json &setup)
{
	if (setup.count("neuron_size") > 0) {
		m_neuron_size = setup["neuron_size"].get<int>();
	}
	if (setup.count("big_capacitor") > 0) {
		m_use_big_capacitor = setup["big_capacitor"].get<bool>();
	}
	if (setup.count("bandwidth") > 0) {
		m_bandwidth = setup["bandwidth"].get<float>();
	}
	if (setup.count("synapse_loss") > 0) {
		m_synapse_loss = setup["synapse_loss"].get<bool>();
	}
	if (setup.count("calib_path") > 0) {
		m_calib_path = setup["calib_path"].get<std::string>();
	}
	if (setup.count("wafer") > 0) {
		m_wafer = setup["wafer"].get<int>();
	}
	if (setup.count("hicann") > 0) {
		m_hicann = setup["hicann"];
	}
	if (setup.count("digital_weight") > 0) {
		m_digital_weight = setup["digital_weight"].get<bool>();
	}
	if (setup.count("ess") > 0) {
		m_ess = setup["ess"].get<bool>();
	}
}

std::vector<PopulationPtr> cypress::BrainScaleS::create_pops(
    ObjectStore &store, const std::vector<cypress::PopulationBase> &populations)
{
	std::vector<PopulationPtr> bs_populations;
	for (size_t i = 0; i < populations.size(); i++) {
		if (populations[i].size() == 0) {
			bs_populations.push_back(nullptr);
			continue;
		}
		if (&(populations[i].type()) == &cypress::SpikeSourceArray::inst()) {
			bs_populations.push_back(::Population::create(
			    store, populations[i].size(), CellType::SpikeSourceArray));
		}
		else if (&populations[i].type() == &cypress::IfCondExp::inst()) {
			bs_populations.push_back(::Population::create(
			    store, populations[i].size(), CellType::IF_cond_exp));
		}
		else if (&populations[i].type() ==
		         &cypress::EifCondExpIsfaIsta::inst()) {
			bs_populations.push_back(
			    ::Population::create(store, populations[i].size(),
			                         CellType::EIF_cond_exp_isfa_ista));
		}
		else {
			throw cypress::NotSupportedException(
			    "Neuron Type not supported by BrainScaleS");
		}
	}
	return bs_populations;
}

template <typename T>
void cypress::BrainScaleS::set_hom_param(T &vec,
                                         const cypress::NeuronParameters &src)
{
	auto &params = vec.parameters();
	for (size_t neuron = 0; neuron < vec.size(); neuron++) {
		set_BS_params(params[neuron], src);
	}
}

template <typename T>
void cypress::BrainScaleS::set_inhom_param(T &vec,
                                           const cypress::PopulationBase &pop)
{
	auto &params = vec.parameters();
	for (size_t neuron = 0; neuron < vec.size(); neuron++) {
		set_BS_params(params[neuron], pop[neuron].parameters());
	}
}

template void cypress::BrainScaleS::set_hom_param<
    TypedCellParameterVector<CellType::IF_cond_exp>>(
    TypedCellParameterVector<CellType::IF_cond_exp> &vec,
    const cypress::NeuronParameters &src);
template void cypress::BrainScaleS::set_hom_param<
    TypedCellParameterVector<CellType::EIF_cond_exp_isfa_ista>>(
    TypedCellParameterVector<CellType::EIF_cond_exp_isfa_ista> &vec,
    const cypress::NeuronParameters &src);

template void cypress::BrainScaleS::set_inhom_param<
    TypedCellParameterVector<CellType::IF_cond_exp>>(
    TypedCellParameterVector<CellType::IF_cond_exp> &vec,
    const cypress::PopulationBase &pop);

template void cypress::BrainScaleS::set_inhom_param<
    TypedCellParameterVector<CellType::EIF_cond_exp_isfa_ista>>(
    TypedCellParameterVector<CellType::EIF_cond_exp_isfa_ista> &vec,
    const cypress::PopulationBase &pop);

void cypress::BrainScaleS::set_population_parameters(
    PopulationPtr &bs_pop, const cypress::PopulationBase &pop)
{
	if (bs_pop == nullptr) {
		return;
	}
	if (&(pop.type()) == &cypress::SpikeSourceArray::inst()) {
		auto &params =
		    reinterpret_cast<
		        TypedCellParameterVector<CellType::SpikeSourceArray> &>(
		        bs_pop->parameters())
		        .parameters();
		for (size_t neuron = 0; neuron < pop.size(); neuron++) {
			params[neuron].spike_times = pop[neuron].parameters().parameters();
		}
		return;
	}

	bool homogeneous = pop.homogeneous_parameters();
	if (&(pop.type()) == &cypress::IfCondExp::inst()) {
		auto &params =
		    reinterpret_cast<TypedCellParameterVector<CellType::IF_cond_exp> &>(
		        bs_pop->parameters());
		const auto &par_names = pop.type().parameter_names;

		for (size_t par = 0; par < par_names.size(); par++) {
			if (homogeneous) {
				set_hom_param(params, pop.parameters());
			}
			else {
				set_inhom_param(params, pop);
			}
		}
	}
	else if (&(pop.type()) == &cypress::EifCondExpIsfaIsta::inst()) {
		auto &params = reinterpret_cast<
		    TypedCellParameterVector<CellType::EIF_cond_exp_isfa_ista> &>(
		    bs_pop->parameters());
		const auto &par_names = pop.type().parameter_names;

		for (size_t par = 0; par < par_names.size(); par++) {
			if (homogeneous) {
				set_hom_param(params, pop.parameters());
			}
			else {
				set_inhom_param(params, pop);
			}
		}
	}
	else {
		throw cypress::NotSupportedException(
		    "Neuron Type not supported by BrainScaleS");
	}
}

bool cypress::BrainScaleS::warn_gsyn_emitted = false;

void cypress::BrainScaleS::set_population_records(
    PopulationPtr &bs_pop, const cypress::PopulationBase &pop)
{
	if (bs_pop == nullptr) {
		return;
	}
	const bool homogeneous_rec = pop.homogeneous_record();
	if (&(pop.type()) == &cypress::SpikeSourceArray::inst()) {
		auto &params =
		    reinterpret_cast<
		        TypedCellParameterVector<CellType::SpikeSourceArray> &>(
		        bs_pop->parameters())
		        .parameters();
		if (homogeneous_rec) {
			if (pop.signals().is_recording(0)) {
				set_hom_rec_spikes(params);
			}
		}
		else {
			set_inhom_rec_spikes(params, pop);
		}
		return;
	}
	auto &params =
	    reinterpret_cast<TypedCellParameterVector<CellType::IF_cond_exp> &>(
	        bs_pop->parameters())
	        .parameters();
	if (homogeneous_rec) {
		set_hom_rec(params, pop);
	}
	else {
		set_inhom_rec(params, pop);
	}
}

boost::shared_ptr<::Connector> cypress::BrainScaleS::get_connector(
    const cypress::ConnectionDescriptor &conn)
{
	if (conn.connector().synapse_name() != "StaticSynapse") {
		throw cypress::ExecutionError(
		    "Only static synapses are supported on "
		    "this backend!");
		// TODO dynamic synapses
	}
	std::string name = conn.connector().name();
	auto &params = conn.connector().synapse()->parameters();
	auto &connector = conn.connector();
	if (name == "AllToAllConnector") {
		return boost::make_shared<::AllToAllConnector>(
		    connector.allow_self_connections(), params[0], params[1]);
	}
	else if (name == "OneToOneConnector") {
		return boost::make_shared<::OneToOneConnector>(
		    connector.allow_self_connections(), params[0], params[1]);
	}
	else if (name == "FixedFanInConnector") {
		return boost::make_shared<::FixedNumberPreConnector>(
		    connector.additional_parameter(),
		    connector.allow_self_connections(), params[0], params[1]);
	}
	else if (name == "FixedFanOutConnector") {
		return boost::make_shared<::FixedNumberPostConnector>(
		    connector.additional_parameter(),
		    connector.allow_self_connections(), params[0], params[1]);
	}
	else if (name == "RandomConnector") {
		return boost::make_shared<::FixedProbabilityConnector>(
		    connector.additional_parameter(),
		    connector.allow_self_connections(), params[0], params[1]);
	}
	return boost::shared_ptr<::Connector>();
}

std::tuple<boost::shared_ptr<::Connector>, boost::shared_ptr<::Connector>>
cypress::BrainScaleS::get_list_connector(
    const cypress::ConnectionDescriptor &conn,
    std::vector<cypress::LocalConnection> &conns_full)
{
	// List connector
	conn.connect(conns_full);
	size_t n_exhs = 0;
	size_t n_inhs = 0;
	for (size_t i = 0; i < conns_full.size(); i++) {
		if (conns_full[i].excitatory()) {
			n_exhs++;
		}
		else if (conns_full[i].inhibitory()) {
			n_inhs++;
		}
		// default = 0 --> ignore!
	}
	ConnectorTypes::vector_type weights(n_exhs);
	ConnectorTypes::vector_type delays(n_exhs);
	auto conns_temp = ::FromListConnector::Connections(n_exhs);

	ConnectorTypes::vector_type weights_inh(n_inhs);
	ConnectorTypes::vector_type delays_inh(n_inhs);
	auto conns_temp_inh = ::FromListConnector::Connections(n_inhs);

	size_t counter_exh = 0, counter_inh = 0;

	for (size_t i = 0; i < conns_full.size(); i++) {
		if (conns_full[i].excitatory()) {
			weights[counter_exh] = conns_full[i].SynapseParameters[0];
			delays[counter_exh] = conns_full[i].SynapseParameters[1];
			conns_temp[counter_exh] = {size_t(conns_full[i].src),
			                           size_t(conns_full[i].tar)};
			counter_exh++;
		}
		else if (conns_full[i].inhibitory()) {
			weights_inh[counter_inh] = -conns_full[i].SynapseParameters[0];
			delays_inh[counter_inh] = conns_full[i].SynapseParameters[1];
			conns_temp_inh[counter_inh] = {size_t(conns_full[i].src),
			                               size_t(conns_full[i].tar)};
			counter_inh++;
		}
	};

	return std::make_tuple(
	    boost::make_shared<::FromListConnector>(std::move(conns_temp), weights,
	                                            delays),
	    boost::make_shared<::FromListConnector>(std::move(conns_temp_inh),
	                                            weights_inh, delays_inh));
}

::PopulationView cypress::BrainScaleS::get_popview(PopulationPtr bs_pop,
                                                   const size_t &start,
                                                   const size_t &end)
{
	if (bs_pop == nullptr) {
		return ::PopulationView();
	}
	if (start == 0 and end == bs_pop->size()) {
		return bs_pop;
	}
	boost::dynamic_bitset<> mask(bs_pop->size());
	for (size_t i = start; i < std::min(end, bs_pop->size()); i++) {
		mask.set(i);
	}
	return ::PopulationView(bs_pop, mask);
}

void cypress::BrainScaleS::manual_placement(
    const cypress::Json &hicann,
    boost::shared_ptr<pymarocco::PyMarocco> marocco,
    std::vector<PopulationPtr> &bs_populations)
{
	if (hicann.is_number()) {
		if (!hicann.is_number_integer()) {
			throw cypress::ExecutionError(
			    "Hicann must be integer or array of integers!");
		}
		for (auto pop : bs_populations) {
			if (pop == nullptr) {
				continue;
			}
			marocco->manual_placement.on_hicann(
			    pop->id(), HMF::Coordinate::HICANNOnWafer(
			                   HMF::Coordinate::Enum(hicann.get<int>())));
		}
	}
	else if (hicann.is_array()) {
		std::vector<HMF::Coordinate::HICANNOnWafer> hicanns;
		for (auto i : hicann) {
			if (!i.is_number_integer()) {
				throw cypress::ExecutionError(
				    "Hicann must be integer or array of integers!");
			}
			hicanns.emplace_back(HMF::Coordinate::HICANNOnWafer(
			    HMF::Coordinate::Enum(i.get<int>())));
		}
		for (auto pop : bs_populations) {
			if (pop == nullptr) {
				continue;
			}
			marocco->manual_placement.on_hicann(pop->id(), hicanns);
		}
	}
	else {
		throw cypress::ExecutionError(
		    "Hicann must be integer or array of integers!");
	}
}

void cypress::BrainScaleS::fetch_data(
    const std::vector<cypress::PopulationBase> &populations,
    const std::vector<PopulationPtr> &bs_populations)
{
	for (size_t i = 0; i < populations.size(); i++) {
		if (populations[i].size() == 0) {
			continue;
		}
		std::vector<std::string> signals = populations[i].type().signal_names;
		for (size_t j = 0; j < signals.size(); j++) {
			bool is_recording = false;
			for (auto neuron : populations[i]) {
				if (neuron.signals().is_recording(j)) {
					is_recording = true;
					break;
				}
			}
			if (is_recording) {
				if (signals[j] == "spikes") {
					auto idx = populations[i][0].type().signal_index("spikes");
					for (size_t k = 0; k < bs_populations[i]->size(); k++) {
						auto neuron = populations[i][k];
						std::vector<float> &spikes =
						    bs_populations[i]->getSpikes(k);
						size_t len = spikes.size();
						if (len == 0) {
							continue;
						}
						auto data =
						    std::make_shared<cypress::Matrix<cypress::Real>>(
						        len, 1);
						for (size_t l = 0; l < len; l++) {
							(*data)(l, 0) = cypress::Real(spikes[l] * 1e3);
						}
						neuron.signals().data(idx.value(), std::move(data));
					}
				}
				else if (signals[j] == "v") {
					auto idx =
					    populations[i][0].type().signal_index(signals[j]);
					for (size_t k = 0; k < bs_populations[i]->size(); k++) {
						auto neuron = populations[i][k];
						auto const &voltageTrace =
						    bs_populations[i]->getMembraneVoltageTrace(k);
						auto len = voltageTrace.size();
						auto data =
						    std::make_shared<cypress::Matrix<cypress::Real>>(
						        len, 2);
						for (size_t l = 0; l < len; l++) {
							(*data)(l, 0) =
							    std::get<0>(voltageTrace[l]);  // time
							(*data)(l, 1) =
							    std::get<1>(voltageTrace[l]);  // mem
						}
						neuron.signals().data(idx.value(), std::move(data));
					}
				}
			}
		}
	}
}
namespace {
inline auto pop_to_bio_neurons(marocco::placement::results::Placement &results,
                               PopulationPtr &source)
{
	std::vector<std::reference_wrapper<const marocco::BioNeuron>> ret;
	for (auto &it : results.find(source->id())) {
		ret.push_back(std::ref(it.bio_neuron()));
	}
	if (ret.size() != source->size()) {
		throw cypress::ExecutionError(
		    "Something wrong in setting low-level weights: resolving neurons "
		    "to bio neurons");
	}
	std::sort(ret.begin(), ret.end(),
	          [](const marocco::BioNeuron &a, const marocco::BioNeuron &b) {
		          return a.neuron_index() < b.neuron_index();
	          });
	return ret;
}

inline auto get_synapse(size_t conn_id, const marocco::BioNeuron &bio_nrn_a,
                        const marocco::BioNeuron &bio_nrn_b,
                        marocco::routing::results::Synapses &results_synapses)
{

	std::vector<
	    std::reference_wrapper<const marocco::routing::results::Synapses::
	                               optional_hardware_synapse_type>>
	    ret;
	/*std::cout << "get_synapse" << std::endl;
	std::cout << "Neuron : " << bio_nrn_a.neuron_index() << " population "
	          << bio_nrn_a.population() << std::endl;
	std::cout << "Neuron : " << bio_nrn_b.neuron_index() << " population "
	          << bio_nrn_b.population() << std::endl;*/
	for (auto &it : results_synapses.find(conn_id, bio_nrn_a, bio_nrn_b)) {
		ret.push_back(std::ref(it.hardware_synapse()));
	}
	if (ret.size() > 1) {
		throw cypress::ExecutionError(
		    "Something went wrong in setting low-level weights: several "
		    "hardware synapses!");
	}
	return ret;
}

void set_low_level_weights_list(
    PopulationPtr &source, PopulationPtr &target, ProjectionPtr &conn_exc,
    ProjectionPtr &conn_inh, std::vector<cypress::LocalConnection> &vec,
    boost::shared_ptr<pymarocco::runtime::Runtime> runtime)
{
	auto &results = runtime->results()->placement;
	auto bio_nrns_a = pop_to_bio_neurons(results, source);
	auto bio_nrns_b = pop_to_bio_neurons(results, target);
	auto &results_synapses = runtime->results()->synapse_routing.synapses();

	for (cypress::LocalConnection &i : vec) {
		if (i.SynapseParameters[0] == 0) {
			continue;  // Weight==0 synapses are deleted
		}
		size_t conn_id;
		uint8_t weight = 0;
		if (i.excitatory()) {
			conn_id = conn_exc->id();
			weight = i.SynapseParameters[0];
		}
		else if (i.inhibitory()) {
			conn_id = conn_inh->id();
			weight = -i.SynapseParameters[0];
		}
		else {
			continue;  // ignore null synapse
		}

		auto syn_hand = get_synapse(conn_id, bio_nrns_a[i.src],
		                            bio_nrns_b[i.tar], results_synapses);
		if (syn_hand.size() == 0) {
			// No synapse found --> probably not mapped to hw/lost
			cypress::global_logger().debug(
			    "cypress",
			    "Ignoring missing synapse in setting low-level weights");
		}
		auto hicann = syn_hand[0].get()->toHICANNOnWafer();
		auto proxy = (*runtime->wafer())[hicann].synapses[*(syn_hand[0].get())];
		proxy.weight = HMF::HICANN::SynapseWeight(weight);
	}
}
}  // namespace
void cypress::BrainScaleS::do_run(cypress::NetworkBase &source,
                                  Real duration) const
{
	cypress::global_logger().info(
	    "cypress",
	    "Running with configuration: neuron size: " +
	        std::to_string(m_neuron_size) +
	        ", big_capacitor: " + (m_use_big_capacitor ? "true" : "false") +
	        ", bandwidth: " + std::to_string(m_bandwidth) + ", synapse loss: " +
	        std::to_string(m_synapse_loss) + ", calib path: " + m_calib_path +
	        ", wafer: " + std::to_string(m_wafer) +
	        ", hicann: " + m_hicann.dump() + ", setting digital weights: " +
	        (m_digital_weight ? "true" : "false"));
	auto start = std::chrono::system_clock::now();
	init_logger();

	ObjectStore store;
	auto marocco = pymarocco::PyMarocco::create();
	marocco->continue_despite_synapse_loss = m_synapse_loss;

	// Choose between Hardware, ESS, and None
	marocco->backend = pymarocco::PyMarocco::Backend::Hardware;

	marocco->calib_backend = pymarocco::PyMarocco::CalibBackend::Binary;
	marocco->neuron_placement.default_neuron_size(
	    m_neuron_size);  // denmems per neuron

	// Some low-level defaults we might consider to change
	marocco->neuron_placement.restrict_rightmost_neuron_blocks(true);  // false
	marocco->neuron_placement.minimize_number_of_sending_repeaters(
	    true);  // false
	marocco->param_trafo.use_big_capacitors =
	    m_use_big_capacitor;  // default true
	marocco->input_placement.consider_firing_rate(true);
	marocco->input_placement.bandwidth_utilization(m_bandwidth);
	marocco->calib_path = m_calib_path;
	// marocco->defects.backend = pymarocco::Defects::Backend::XML;
	marocco->defects.path = m_calib_path;
	marocco->default_wafer = HMF::Coordinate::Wafer(m_wafer);

	auto runtime =
	    pymarocco::runtime::Runtime::create(HMF::Coordinate::Wafer(m_wafer));
	// Save marocco to ObjectStore
	ObjectStore::Settings settings;
	ObjectStore::metadata_map metadata;
	metadata["marocco"] = marocco;
	metadata["marocco_runtime"] = runtime;
	store.setup(settings, metadata);  // runtime object

	// Create populations
	const std::vector<PopulationBase> &populations = source.populations();
	std::vector<PopulationPtr> bs_populations = create_pops(store, populations);

	for (size_t i = 0; i < populations.size(); i++) {
		set_population_parameters(bs_populations[i], populations[i]);
		set_population_records(bs_populations[i], populations[i]);
	}

	// Random generator used for random connectors
	boost::shared_ptr<RandomGenerator> rng =
	    boost::make_shared<NativeRandomGenerator>(1234);

	std::vector<ProjectionPtr> projections;
	std::vector<ProjectionPtr> list_projections_exc;
	std::vector<ProjectionPtr> list_projections_inh;
	std::vector<std::vector<cypress::LocalConnection>> list_connections;

	for (size_t i = 0; i < source.connections().size(); i++) {
		auto conn = source.connections()[i];
		auto source = get_popview(bs_populations[conn.pid_src()],
		                          conn.nid_src0(), conn.nid_src1());
		auto target = get_popview(bs_populations[conn.pid_tar()],
		                          conn.nid_tar0(), conn.nid_tar1());
		std::string recep_type = conn.connector().synapse()->parameters()[0] > 0
		                             ? "excitatory"
		                             : "inhibitory";

		// TODO dynamic synapses
		auto connect = get_connector(conn);
		if (connect) {
			projections.emplace_back(
			    ::Projection::create(store, source, target, connect, rng, "",
			                         recep_type));  //+ Synapse_dynamics , label
		}
		else {
			projections.emplace_back(ProjectionPtr());
			list_connections.emplace_back();
			auto tuple = get_list_connector(conn, list_connections.back());
			list_projections_exc.emplace_back(
			    ::Projection::create(store, source, target, std::get<0>(tuple),
			                         rng, "", "excitatory"));
			list_projections_inh.emplace_back(
			    ::Projection::create(store, source, target, std::get<1>(tuple),
			                         rng, "", "inhibitory"));
		}
	}

	// ProjectionMatrix weights = proj->getWeights(); Used to get the set
	// weights

	manual_placement(m_hicann, marocco, bs_populations);

	// Run mapping, necessary for changing low-level parameters
	marocco->backend = pymarocco::PyMarocco::Backend::None;
	marocco->skip_mapping = false;
	store.run(duration);  // ms
	submit(store);
	set_stahl_params(runtime->wafer(), 1023, 2);

	if (m_ess) {
		marocco->backend = pymarocco::PyMarocco::Backend::ESS;
	}
	else {
		marocco->backend = pymarocco::PyMarocco::Backend::Hardware;
	}
	marocco->skip_mapping = true;

	// Set low-level weights
	if (m_digital_weight) {
		size_t counter = 0;
		for (size_t i = 0; i < projections.size(); i++) {
			auto conn = source.connections()[i];
			if (!projections[i]) {
				set_low_level_weights_list(bs_populations[conn.pid_src()],
				                           bs_populations[conn.pid_tar()],
				                           list_projections_exc[counter],
				                           list_projections_inh[counter],
				                           list_connections[counter], runtime);
				counter++;
			}
			else {
				auto hw_synapses =
				    runtime->results()->synapse_routing.synapses().find(
				        projections[i]->id());
				for (auto hw_syn : hw_synapses) {
					auto syn_hand = hw_syn.hardware_synapse();
					auto hicann = syn_hand->toHICANNOnWafer();
					auto proxy =
					    (*runtime->wafer())[hicann].synapses[*syn_hand];
					proxy.weight = HMF::HICANN::SynapseWeight(
					    conn.connector().synapse()->parameters()[0]);
				}
			}
		}
	}

	// Run the emulation
	auto buildconn = std::chrono::system_clock::now();
	store.run(duration);
	submit(store);
	auto execrun = std::chrono::system_clock::now();
	fetch_data(populations, bs_populations);

	store.reset();

	auto finished = std::chrono::system_clock::now();
	source.runtime({std::chrono::duration<Real>(finished - start).count(),
	                std::chrono::duration<Real>(execrun - buildconn).count(),
	                std::chrono::duration<Real>(buildconn - start).count(),
	                std::chrono::duration<Real>(finished - execrun).count()});
}

std::unordered_set<const cypress::NeuronType *>
cypress::BrainScaleS::supported_neuron_types() const
{
	return std::unordered_set<const cypress::NeuronType *>{
	    &cypress::SpikeSourceArray::inst(), &cypress::IfCondExp::inst(),
	    &cypress::EifCondExpIsfaIsta::inst()};
}
